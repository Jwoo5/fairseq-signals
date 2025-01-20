import torch
import torch.nn as nn

from dataclasses import dataclass, field
from omegaconf import MISSING, II

from fairseq_signals.models import register_model
from fairseq_signals.utils import utils
from fairseq_signals.models.ecg_transformer import (
    TransformerConfig,
    TransformerModel,
)
from fairseq_signals.tasks import Task
from fairseq_signals.modules import LayerNorm

@dataclass
class BlindTransformerQAConfig(TransformerConfig):
    # encode text
    vocab_size: int = field(
        default=30522,
        metadata={
            "help": "vocab size of tokenizer. "
            "default is set for BertTokenizer from huggingface (bert-base-uncased)"
        }
    )
    load_bert_embedding: bool = field(
        default=True,
        metadata={
            "help": "whether to load bert embedding parameters to encode texts"
        }
    )
    pad_token: int = II("task.pad_token")

    # output
    num_labels: int = field(
        default=MISSING,
        metadata={
            "help": "number of classes in the dataset"
        }
    )

@register_model("blind_transformer_qa", dataclass=BlindTransformerQAConfig)
class BlindTransformerQAModel(TransformerModel):
    def __init__(self, cfg: BlindTransformerQAConfig):
        super().__init__(cfg)

        self.layer_norm = LayerNorm(cfg.encoder_embed_dim)

        if cfg.load_bert_embedding:
            from transformers import AutoModel
            bert_embeddings = AutoModel.from_pretrained("bert-base-uncased").embeddings
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
        self.vocab_size = cfg.vocab_size

        self.proj = nn.Linear(cfg.encoder_embed_dim, cfg.num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

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

    def get_logits(self, net_output, normalize=False, **kwargs):
        logits = net_output["x"]
        
        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)
        
        return logits

    def get_targets(self, sample, net_output, **kwargs):
        return sample["answer"].float()

    def forward(
        self,
        text,
        text_padding_mask=None,
        **kwargs
    ):
        x = self.language_embedding(text)
        x = self.layer_norm(x)
        x = self.dropout_input(x)
        
        x_pos = (
            self.position_embedding(
                torch.arange(x.size(1), device=x.device).repeat((x.size(0), 1))
            )
        )
        x += x_pos
        
        x_type = self.token_type_embedding(x.new_zeros(x.shape[:-1], dtype=int))
        x += x_type
        
        x = self.encoder(x, padding_mask=text_padding_mask)["x"]

        if text_padding_mask is not None and text_padding_mask.any():
            x[text_padding_mask] = 0
        
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))
        
        x = self.proj(x)
        
        return {"x": x, "padding_mask": text_padding_mask}