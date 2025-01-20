import contextlib
from dataclasses import dataclass, field
from typing import Any
import logging
from omegaconf import II

import torch
import torch.nn as nn

from fairseq_signals.models.transformer import (
    TransformerConfig,
    TransformerModel,
    TransformerFinetuningConfig,
    TransformerFinetuningModel
)
from fairseq_signals.tasks import Task
from fairseq_signals.modules import (
    GradMultiply,
    LayerNorm,
    ConvFeatureExtraction,
    ConvPositionalEncoding,
)

logger = logging.getLogger(__name__)

@dataclass
class ECGLanguageTransformerConfig(TransformerConfig):
    # configs for convnets
    extractor_mode: str  = field (
        default = "default",
        metadata = {
            "help": "mode for conv feature extractor. 'default' has a single group norm with d"
            "groups in the first conv block, whereas 'layer_norm' has layer norms in "
            "every block (meant to use with normalize = True)"
        }
    )
    conv_feature_layers: str = field(
        default="[(256, 2, 2)] * 4",
        metadata={
            "help": "string describing convolutional feature extraction layers "
            "in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    in_d: int = field(
        default=12,
        metadata={"help": "input dimension of ECGs"}
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )

    # positional embeddings for ecg
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embeddings"},
    )


    # configs for embedding layer (for languages)
    vocab_size: int = field(
        default=30522,
        metadata={
            'help': 'vocab size of tokenizer.'
            'default is for BertTokenizer from huggingface (bert-base-uncased)'
        }
    )
    load_bert_embedding: bool = field(
        default=True,
        metadata={
            'help': 'whether to load bert embedding parameters to encode texts'
        }
    )
    pad_token: int = II("task.pad_token")
    sep_token: int = II("task.sep_token")
    max_text_size: int = II('task.max_text_size')

    #XXX to be removed
    normalize: bool = II('task.normalize')
    data: str = II('task.data')
    args: Any = None

class ECGLanguageTransformerModel(TransformerModel):
    def __init__(self, cfg: ECGLanguageTransformerConfig):
        super().__init__(cfg)
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed_dim = feature_enc_layers[-1][0]
        self.encoder_embed_dim = cfg.encoder_embed_dim

        self.feature_extractor = ConvFeatureExtraction(
            conv_layers=feature_enc_layers,
            in_d=cfg.in_d,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias
        )

        self.post_extract_proj = (
            nn.Linear(self.embed_dim, cfg.encoder_embed_dim)
            if self.embed_dim != cfg.encoder_embed_dim
            else None
        )

        self.feature_grad_mult = cfg.feature_grad_mult
        self.conv_pos = ConvPositionalEncoding(cfg)
        self.feats_layer_norm = LayerNorm(self.embed_dim)
        self.layer_norm = LayerNorm(cfg.encoder_embed_dim)

        if cfg.load_bert_embedding:
            from transformers import AutoModel
            bert_embeddings = AutoModel.from_pretrained('bert-base-uncased').embeddings
            #XXX to be changed to self.word_embeddings, ...
            self.language_embedding = bert_embeddings.word_embeddings
            self.position_embedding = bert_embeddings.position_embeddings
            self.token_type_embedding = bert_embeddings.token_type_embeddings
        else:
            self.language_embedding = nn.Embedding(
                cfg.vocab_size, cfg.encoder_embed_dim, padding_idx=cfg.pad_token
            )
            self.position_embedding = nn.Embedding(
                cfg.max_text_size, cfg.encoder_embed_dim
            )
            self.token_type_embedding = nn.Embedding(
                2, cfg.encoder_embed_dim
            )

        self.pad_token = cfg.pad_token
        self.sep_token = cfg.sep_token
        self.special_tokens = [cfg.pad_token, cfg.sep_token]
        self.vocab_size = cfg.vocab_size

        self.num_updates = 0

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrate a (possibly old) state dict for new versions."""
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)
        
        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2])
        
        return input_lengths.to(torch.long)
    
    def forward(
        self,
        ecg,
        text,
        ecg_padding_mask=None,
        text_padding_mask=None,
        ecg_2=None,
        ecg_2_padding_mask=None,
        **kwargs
    ):
        ecg_features_2 = None
        if self.feature_grad_mult > 0:
            ecg_features = self.feature_extractor(ecg)
            if ecg_2 is not None:
                ecg_features_2 = self.feature_extractor(ecg_2)
            if self.feature_grad_mult != 1.0:
                ecg_features = GradMultiply.apply(ecg_features, self.feature_grad_mult)
                if ecg_features_2 is not None:
                    ecg_features_2 = GradMultiply.apply(ecg_features_2, self.feature_grad_mult)
        else:
            with torch.no_grad():
                ecg_features = self.feature_extractor(ecg)
                if ecg_features_2 is not None:
                    ecg_features_2 = self.feature_extractor(ecg_2)

        ecg_features = ecg_features.transpose(1,2)
        ecg_features = self.feats_layer_norm(ecg_features)

        if ecg_padding_mask is not None and ecg_padding_mask.any():
            input_lengths = (1 - ecg_padding_mask.long()).sum(-1)
            if input_lengths.dim() > 1:
                for input_len in input_lengths:
                    assert (input_len == input_len[0]).all()
                input_lengths = input_lengths[:, 0]
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            ecg_padding_mask = torch.zeros(
                ecg_features.shape[:2], dtype=ecg_features.dtype, device=ecg_features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            ecg_padding_mask[
                (
                    torch.arange(ecg_padding_mask.shape[0], device=ecg_padding_mask.device),
                    output_lengths - 1
                )
            ] = 1
            ecg_padding_mask = (1 - ecg_padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            ecg_padding_mask = None

        if ecg_features_2 is not None:
            ecg_features_2 = ecg_features_2.transpose(1, 2)
            ecg_features_2 = self.feats_layer_norm(ecg_features_2)

        if ecg_2_padding_mask is not None and ecg_2_padding_mask.any():
            input_lengths = (1 - ecg_2_padding_mask.long()).sum(-1)
            if input_lengths.dim() > 1:
                for input_len in input_lengths:
                    assert (input_len == input_len[0]).all()
                input_lengths = input_lengths[:, 0]
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            ecg_2_padding_mask = torch.zeros(
                ecg_features_2.shape[:2], dtype=ecg_features_2.dtype, device=ecg_features_2.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            ecg_2_padding_mask[
                (
                    torch.arange(ecg_2_padding_mask.shape[0], device=ecg_2_padding_mask.device),
                    output_lengths - 1
                )
            ] = 1
            ecg_2_padding_mask[torch.where(output_lengths == 0)] = 0
            ecg_2_padding_mask = (1 - ecg_2_padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            ecg_2_padding_mask = None

        if text_padding_mask is not None and not text_padding_mask.any():
            text_padding_mask = None

        if self.post_extract_proj is not None:
            ecg_features = self.post_extract_proj(ecg_features)
            if ecg_features_2 is not None:
                ecg_features_2 = self.post_extract_proj(ecg_features_2)

        ecg_features = self.dropout_input(ecg_features)
        if self.cfg.apply_mask and self.training:
            ecg_features, _ = self.apply_mask(
                ecg_features,
                ecg_padding_mask,
                mask_indices=None
            )
        
        if ecg_features_2 is not None:
            ecg_features_2 = self.dropout_input(ecg_features_2)
            if self.cfg.apply_mask and self.training:
                ecg_features_2, _ = self.apply_mask(
                    ecg_features_2,
                    ecg_2_padding_mask,
                    mask_indices=None
                )
            sep_token_embeddings = self.language_embedding(
                torch.full((len(ecg_features), 1), fill_value=self.sep_token, device=ecg_features.device)
            )
            ecg_features_2 = torch.cat(
                [sep_token_embeddings, ecg_features_2], dim=1
            )

            if ecg_padding_mask is None and ecg_2_padding_mask is not None:
                ecg_padding_mask = ecg_features.new_zeros(ecg_features.shape[:2], dtype=bool)
            elif ecg_padding_mask is not None and ecg_2_padding_mask is None:
                ecg_2_padding_mask = ecg_features_2.new_zeros(ecg_features_2.shape[:2], dtype=bool)

            if ecg_padding_mask is not None and ecg_2_padding_mask is not None:
                sep_padding_mask = ecg_2_padding_mask.new_zeros((len(ecg_2_padding_mask), 1,))
                sep_padding_mask[torch.where(ecg_2_padding_mask.all(dim=-1))] = True
                ecg_padding_mask = torch.cat(
                    [
                        ecg_padding_mask,
                        sep_padding_mask,
                        ecg_2_padding_mask
                    ], dim=1
                )

            ecg_features = torch.cat(
                [ecg_features, ecg_features_2], dim=1
            )

        ecg_features_conv = self.conv_pos(ecg_features, channel_first=False)
        ecg_features = ecg_features + ecg_features_conv
        ecg_features_type_embedding = (
            self.token_type_embedding(
                ecg_features.new_zeros(ecg_features.shape[:-1], dtype=int)
            )
        )
        ecg_features = ecg_features + ecg_features_type_embedding

        text_features = self.language_embedding(text)
        text_features = self.layer_norm(text_features)
        text_features = self.dropout_input(text_features)

        text_features_pos = self.position_embedding(
            torch.arange(
                text_features.size(1),
                device=text_features.device
            ).repeat((text_features.size(0), 1))
        )
        text_features = text_features + text_features_pos
        text_features_type_embedding = (
            self.token_type_embedding(
                text_features.new_ones(text_features.shape[:-1], dtype=int)
            )
        )
        text_features = text_features + text_features_type_embedding

        x = torch.cat([ecg_features, text_features], dim=1)
        if ecg_padding_mask is not None or text_padding_mask is not None:
            ecg_padding_mask = (
                ecg_features.new_zeros(ecg_features.shape[:-1], dtype=bool)
                if ecg_padding_mask is None else ecg_padding_mask
            )
            text_padding_mask = (
                text_features.new_zeros(text_features.shape[:-1], dtype=bool)
                if text_padding_mask is None else text_padding_mask
            )

            padding_mask = torch.cat([ecg_padding_mask, text_padding_mask], dim=1)
        else:
            padding_mask = None

        res = self.encoder(x, padding_mask=padding_mask)
        x = res["x"]
        saliency = res["saliency"]

        return {"x": x, "padding_mask": padding_mask, "saliency": saliency}

    def extract_features(
        self,
        ecg,
        text,
        ecg_padding_mask,
        text_padding_mask,
        ecg_2,
        ecg_2_padding_mask
    ):
        res = self.forward(
            ecg=ecg,
            text=text,
            ecg_padding_mask=ecg_padding_mask,
            text_apdding_mask=text_padding_mask,
            ecg_2=ecg_2,
            ecg_2_padding_mask=ecg_2_padding_mask
        )
        return res

    def get_logits(self, net_output, **kwargs):
        raise NotImplementedError()

    def get_targets(self, sample, net_output, **kwargs):
        raise NotImplementedError()
    
    @classmethod
    def from_pretrained(
        cls,
        model_path,
        cfg: ECGLanguageTransformerConfig,
        **kwargs,
    ):
        """
        Load a :class:`~fairseq_signals.models.ECGLanguageTransformerModel` from a pre-trained model
        checkpoint.

        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (ECGLanguageTransformerConfig): cfg to override some arguments of pre-trained model
        """

        arg_overrides = {
            "feature_grad_mult": cfg.feature_grad_mult,
            "attn_mask_type": "bidirectional",
            "load_bert_embedding": False,
        }

        assert cfg.sep_token == cfg.args.task.sep_token, (
            "Special token [SEP] is different between pre-training and here."
            "Please check that --sep_token is the same for both pre-training and here"
        )
        assert cfg.pad_token == cfg.args.task.pad_token, (
            "Special token [PAD] id is different between pre-training and here."
            "Please check that --pad_token is the same for both pre-training and here"
        )

        model = super().from_pretrained(model_path, cfg, arg_overrides)

        return model

@dataclass
class ECGLanguageTransformerFinetuningConfig(TransformerFinetuningConfig, ECGLanguageTransformerConfig):
    # overriding arguments
    feature_grad_mult: float = 1.0

class ECGLanguageTransformerFinetuningModel(TransformerFinetuningModel):
    def __init__(self, cfg: ECGLanguageTransformerFinetuningConfig, encoder: ECGLanguageTransformerModel):
        super().__init__(cfg, encoder)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: ECGLanguageTransformerFinetuningConfig, task: Task):
        """Build a new model instance."""
        if cfg.model_path and not cfg.no_pretrained_weights:
            encoder = ECGLanguageTransformerModel.from_pretrained(cfg.model_path, cfg)
        else:
            encoder = ECGLanguageTransformerModel(cfg)
        return cls(cfg, encoder)

    def get_logits(self, net_output, normalize=False, aggregate=False, **kwargs):
        return NotImplementedError()
    
    def get_targets(self, sample, net_output, **kwargs):
        raise NotImplementedError()

    def forward(
        self,
        ecg,
        text,
        ecg_padding_mask=None,
        text_padding_mask=None,
        ecg_2=None,
        ecg_2_padding_mask=None,
        **kwargs
    ):
        args = {
            "ecg": ecg,
            "text": text,
            "ecg_padding_mask": ecg_padding_mask,
            "text_padding_mask": text_padding_mask,
            "ecg_2": ecg_2,
            "ecg_2_padding_mask": ecg_2_padding_mask
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.encoder.extract_features(**args)

        return res