import contextlib
from dataclasses import dataclass, field
from omegaconf import MISSING, II
from typing import cast
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from fairseq_signals import tasks
from fairseq_signals.utils import utils
from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_signals.models.pretraining_model import PretrainingConfig, PretrainingModel
from fairseq_signals.models.finetuning_model import FinetuningConfig, FinetuningModel
from fairseq_signals.tasks import Task

logger = logging.getLogger(__name__)

@dataclass
class LSTMConfig(PretrainingConfig):
    lstm_input_dim: int = field(
        default=768,
        metadata={
        "help": "the number of expected features in the input x for the lstm model"
        }
    )
    lstm_embed_dim: int = field(
        default=512,
        metadata={"help": "hidden dimension for the lstm model"}
    )
    lstm_num_layer: int = field(
        default=2,
        metadata={"help": "number of hidden layers of lstm"}
    )
    batch_first: bool = field(
        default=True,
        metadata={
            "help": "if True, then the input and output tensors are provided as (batch, seq, feature) "
            "instead of (seq, batch, feature)."
        }
    )
    dropout: float = field(
        default=0,
        metadata={
            "help": "if non-zero, introduces a Dropout layer on the outputs of each LSTM layer "
            "except the last layer, with dropout probability"
        }        
    )
    bidirectional: bool = field(
        default=False,
        metadata={
            "help": "if True, becomes a bidirectional LSTM"
        }
    )

class LSTMModel(PretrainingModel):
    def __init__(self, cfg: LSTMConfig):
        super().__init__(cfg)

        self.batch_first = cfg.batch_first

        self.encoder = nn.LSTM(
            input_size=cfg.lstm_input_dim,
            hidden_size=cfg.lstm_embed_dim,
            num_layers=cfg.lstm_num_layer,
            batch_first=cfg.batch_first,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional
        )

        self.num_updates = 0
    
    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def forward(self, x, padding_mask=None, **kwargs):
        raise NotImplementedError()

    def extract_features(self, source, padding_mask):
        raise NotImplementedError()
    
    @classmethod
    def from_pretrained(
        cls,
        model_path,
        cfg: LSTMConfig,
        arg_appended=None,
        **kwargs
    ):
        """
        Load a :class:`~fairseq_signals.models.LSTMModel` from a pre-trained model checkpoint

        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (LSTMConfig): cfg to override some arguments of pre-trained model
        """

        arg_overrides = {
            "dropout": cfg.dropout,
        }
        if arg_appended is not None:
            arg_overrides.update(arg_appended)

        state = checkpoint_utils.load_checkpoint_to_cpu(model_path, arg_overrides)
        args = state.get("cfg", None)
        if args is None:
            args = convert_namespace_to_omegaconf(state["args"])
        args.criterion = None
        args.lr_scheduler = None
        cfg.args = args

        assert cfg.normalize == args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )
        assert cfg.filter == args.task.filter, (
            "Fine-tuning works best when signal filtering for data is the same. "
            "Please check that --filter is set or unset for both pre-training and here"
        )

        args.task.data = cfg.data
        task = tasks.setup_task(args.task, from_checkpoint=True)
        model = task.build_model(args.model)

        if hasattr(model, "remove_pretraining_modules"):
            model.remove_pretraining_modules()
        
        model.load_state_dict(state["model"], strict=True)
        logger.info(f"Loaded pre-trained model parameters from {model_path}")

        return model

@dataclass
class LanguageLSTMConfig(LSTMConfig):
    vocab_size: int = field(
        default=30522,
        metadata={
            "help": "vocab size of tokenizer. "
            "default is aligned with BertTokenizer from huggingface (bert-base-uncased)"
        }
    )
    load_bert_embedding: bool = field(
        default=True,
        metadata={
            "help": "whether to load bert embedding parameters to encode texts. "
            "if true, it automatically loads weight from huggingface's bert-base-uncased embeddings"
        }
    )

    max_text_size: int = II("task.max_text_size")

class LanguageLSTMModel(LSTMModel):
    def __init__(self, cfg: LanguageLSTMConfig):
        super().__init__(cfg)
        
        if cfg.load_bert_embedding:
            from transformers import AutoModel
            bert_embeddings = AutoModel.from_pretrained("bert-base-uncased").embeddings
            self.word_embeddings = bert_embeddings.word_embeddings
        else:
            self.word_embeddings = nn.Embedding(
                cfg.vocab_size, cfg.lstm_input_dim, padding_idx=cfg.pad_token
            )
        
        self.num_updates = 0

    def forward(self, x, padding_mask=None, **kwargs):
        x = self.word_embeddings(x)
        bsz, tsz, csz = x.shape
        
        if not self.batch_first:
            x = x.view(tsz, bsz, csz)
        
        if padding_mask is not None:
            lengths = (tsz - padding_mask.sum(dim=1)).tolist()
            x = pack_padded_sequence(x, lengths, batch_first=self.batch_first, enforce_sorted=False)

        _, (h_n, c_n) = self.encoder(x)
        x = torch.cat(
            [h_n.transpose(0, 1).flatten(1), c_n.transpose(0, 1).flatten(1)], dim=1
        )

        return {"x": x}

    def extract_features(self, source, padding_mask):
        return self.forward(source, padding_mask)
    
    @classmethod
    def from_pretrained(
        cls,
        model_path,
        cfg: LanguageLSTMConfig
    ):
        """
        Load a :class:`~fairseq_signals.models.LanguageLSTMModel` from a pre-trained model checkpoint

        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (LanguageLSTMConfig): cfg to override some arguments of pre-trained model
        """

        arg_overrides = {
            "load_bert_embedding": False
        }

        model = super().from_pretrained(model_path, cfg, arg_overrides)

        assert cfg.sep_token == cfg.args.task.sep_token, (
            "Special token [SEP] is different between pre-training and here."
            "Please check that --sep_token is the same for both pre-training and here"
        )
        assert cfg.pad_token == cfg.args.task.pad_token, (
            "Special token [PAD] id is different between pre-training and here."
            "Please check that --pad_token is the same for both pre-training and here"
        )

        return model