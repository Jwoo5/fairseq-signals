import math
import contextlib
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Optional
import logging
from omegaconf import II

import numpy as np
import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertModel,
    BertPredictionHeadTransform
)
from .bert_model import BertCrossLayer

from fairseq_signals import tasks
from fairseq_signals.tasks import Task
from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_signals.models import register_model
from fairseq_signals.models.ecg_transformer import ECGTransformerConfig, ECGTransformerModel
from fairseq_signals.models.pretraining_model import PretrainingModel
from fairseq_signals.models.finetuning_model import FinetuningConfig, FinetuningModel

logger = logging.getLogger(__name__)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

@dataclass
class M3AEConfig(ECGTransformerConfig):
    # ecg encoder
    load_pretrained_weights: bool = field(
        default=True,
        metadata={
            "help": "whether to load pretrained weights for ecg encoder"
        }
    )
    pretrained_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to the pretrained ecg encoder"
        }
    )

    # text
    vocab_size: int = field(
        default=30522,
        metadata={
            'help': 'vocab size of tokenizer.'
            'default is set from BertTokenizer from huggingface (bert-base-uncased)'
        }
    )

    # multi-modal fusion transformer
    hidden_dim: int = field(
        default=768,
        metadata={
            "help": "hidden dimension size of the multi-modal fusion transformer"
        }
    )
    num_layers: int = field(
        default=6,
        metadata={
            "help": "num layers in the multi-modal fusion transformer"
        }
    )
    num_heads: int = field(
        default=12,
        metadata={
            "help": "num attention heads in the multi-modal fusion transformer"
        }
    )
    drop_rate: float = field(
        default=0.1,
        metadata={
            "help": "dropout probability in the multi-modal fusion transformer"
        }
    )
    num_top_layer: int = field(
        default=6,
        metadata={
            "help": "num top layers in the multi-modal fusion transformer"
        }
    )
    mim_layer: int = field(
        default=3,
        metadata={
            "help": "layer index to perform MIM"
        }
    )

    # mask
    mim_prob: float = field(
        default=0.75,
        metadata={
            "help": "mask ratio for ecg"
        }
    )
    mim_decoder_hidden_dim: int = field(
        default=384,
        metadata={
            "help": "hidden dimension size of the mim decoder layer"
        }
    )
    mim_decoder_num_layers: int = field(
        default=4,
        metadata={
            "help": "num layers in the mim decoder layer"
        }
    )
    mim_decoder_num_heads: int = field(
        default=6,
        metadata={
            "help": "num attention heads in the mim decoder layer"
        }
    )

    max_text_size: int = II('task.max_text_size')

@register_model("m3ae", dataclass=M3AEConfig)
class M3AEModel(PretrainingModel):
    """
    Model implementation of M3AE (https://arxiv.org/abs/2209.07098).
    The official implementation is provided from (https://github.com/zhjohnchan/M3AE).
    """
    def __init__(self, cfg: M3AEConfig):
        super().__init__(cfg)
        self.cfg = cfg

        self.vocab_size = cfg.vocab_size

        self.mim_prob = cfg.mim_prob
        self.mim_layer = cfg.mim_layer

        if cfg.load_pretrained_weights and cfg.pretrained_model_path is not None:
            self.ecg_encoder = ECGTransformerModel.from_pretrained(cfg.pretrained_model_path, cfg)
        else:
            self.ecg_encoder = ECGTransformerModel(cfg)
        self.class_embedding = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.language_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.language_encoder.pooler = None

        self.multi_modal_language_proj = nn.Linear(cfg.encoder_embed_dim, cfg.hidden_dim)
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_ecg_proj = nn.Linear(cfg.encoder_embed_dim, cfg.hidden_dim)
        self.multi_modal_ecg_proj.apply(init_weights)
        
        self.modality_type_embeddings = nn.Embedding(2, cfg.hidden_dim)
        self.modality_type_embeddings.apply(init_weights)
        
        bert_config = BertConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_dim,
            num_hidden_layers=cfg.num_layers,
            num_attention_heads=cfg.num_heads,
            intermediate_size=cfg.hidden_dim * 4,
            max_position_embeddings=cfg.max_text_size,
            hidden_dropout_prob=cfg.drop_rate,
            attention_probs_dropout_prob=cfg.drop_rate
        )
        self.multi_modal_ecg_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(cfg.num_top_layer)]
        )
        self.multi_modal_ecg_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(cfg.num_top_layer)]
        )
        self.multi_modal_language_layers.apply(init_weights)
    
        self.multi_modal_ecg_pooler = Pooler(cfg.hidden_dim)
        self.multi_modal_ecg_pooler.apply(init_weights)
        self.multi_modal_language_pooler = Pooler(cfg.hidden_dim)
        self.multi_modal_language_pooler.apply(init_weights)
    
        self.mlm_head = MLMHead(bert_config)
        self.mlm_head.apply(init_weights)
        self.mim_head = MIMHead(cfg)
        self.mim_head.apply(init_weights)
        self.itm_head = ITMHead(cfg.hidden_dim * 2)
        self.itm_head.apply(init_weights)
        
    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def random_masking(self, x, mask_ratio):
        x_ = x[:, :1]
        x = x[:, 1:]
        
        bsz, tsz, csz = x.shape
        len_keep = int(tsz * (1 - mask_ratio))
        
        noise = torch.rand(bsz, tsz, device=x.device)
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, csz))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([bsz, tsz], device=x.device)
        mask[:, :len_keep] = 0
        
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # append cls token
        x_masked = torch.cat((x_, x_masked), dim=1)
        
        return x_masked, mask, ids_restore

    def forward(
        self,
        ecg,
        text,
        ecg_padding_mask,
        text_attention_mask,
        ecg_2=None,
        ecg_2_padding_mask=None,
        mask=True,
        features_only=False,
        **kwargs
    ):
        if ecg_padding_mask is not None and not ecg_padding_mask.any():
            ecg_padding_mask = None

        assert ecg_padding_mask is None, (
            "all the ecgs in a batch should have the same size for M3AE model."
        )

        ret = dict()

        uni_modal_text_feats = self.language_encoder.embeddings(input_ids=text)
        text_input_shape = text_attention_mask.size()
        extended_text_masks = self.language_encoder.get_extended_attention_mask(text_attention_mask, text_input_shape)
        for layer in self.language_encoder.encoder.layer:
            uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
        uni_modal_text_feats = self.multi_modal_language_proj(uni_modal_text_feats)

        uni_modal_ecg_feats, ecg_padding_mask = (
            self.ecg_encoder.get_embeddings(ecg, padding_mask=ecg_padding_mask)
        )
        if ecg_2 is not None:
            assert hasattr(self, "sep_embedding"), (
                "you should initialize `sep_embedding` for processing more than one ecg"
            )
            bsz, tsz = uni_modal_ecg_feats.size(0), uni_modal_ecg_feats.size(1)
            uni_modal_ecg_feats_2, ecg_2_padding_mask = (
                self.ecg_encoder.get_embeddings(ecg_2, padding_mask=ecg_2_padding_mask)
            )
            sep_emb = self.sep_embedding.repeat((len(uni_modal_ecg_feats_2), 1, 1))
            uni_modal_ecg_feats_2 = torch.cat([sep_emb, uni_modal_ecg_feats_2], dim=1)
            uni_modal_ecg_feats = torch.cat([uni_modal_ecg_feats, uni_modal_ecg_feats_2], dim=1)

            if ecg_2_padding_mask is not None and ecg_2_padding_mask.any():
                sep_padding_mask = ecg_2_padding_mask.new_zeros((len(ecg_2_padding_mask), 1,))
                sep_padding_mask[torch.where(ecg_2_padding_mask.all(dim=-1))] = True
                ecg_1_padding_mask = ecg_2_padding_mask.new_zeros((bsz, tsz))
                ecg_padding_mask = torch.cat(
                    [
                        ecg_1_padding_mask.new_zeros((len(ecg_1_padding_mask), 1)),
                        ecg_1_padding_mask,
                        sep_padding_mask,
                        ecg_2_padding_mask
                    ], dim=1
                )
            else:
                ecg_padding_mask = None
        cls_emb = self.class_embedding.repeat((len(uni_modal_ecg_feats), 1, 1))
        uni_modal_ecg_feats = torch.cat([cls_emb, uni_modal_ecg_feats], dim=1)

        if mask:
            uni_modal_ecg_feats, mim_masks, mim_ids_restore = self.random_masking(
                uni_modal_ecg_feats, self.mim_prob
            )
            ecg_result = self.ecg_encoder.get_output(uni_modal_ecg_feats)
            ret["mim_masks"] = mim_masks
            ret["mim_ids_restore"] = mim_ids_restore
        else:
            ecg_result = self.ecg_encoder.get_output(uni_modal_ecg_feats, ecg_padding_mask)
        uni_modal_ecg_feats = ecg_result["x"]
        uni_modal_ecg_feats = self.multi_modal_ecg_proj(uni_modal_ecg_feats)

        if ecg_padding_mask is not None and ecg_padding_mask.any():
            ecg_attention_mask = ~ecg_padding_mask
        else:
            ecg_attention_mask = torch.ones(
                (uni_modal_ecg_feats.size(0), uni_modal_ecg_feats.size(1)),
                dtype=torch.long,
                device=ecg.device
            )
        extended_ecg_masks = (
            self.language_encoder.get_extended_attention_mask(ecg_attention_mask, ecg_attention_mask.size())
        )

        uni_modal_text_feats, uni_modal_ecg_feats = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(text, dtype=int)),
            uni_modal_ecg_feats + self.modality_type_embeddings(torch.ones_like(ecg_attention_mask, dtype=int))
        )

        x, y = uni_modal_text_feats, uni_modal_ecg_feats
        for layer_idx, (text_layer, ecg_layer) in enumerate(
            zip(self.multi_modal_language_layers, self.multi_modal_ecg_layers)
        ):
            if mask and self.mim_layer == layer_idx:
                (
                    ret[f"multi_modal_text_feats_{layer_idx}"],
                    ret[f"multi_modal_ecg_feats_{layer_idx}"]
                ) = x, y
            
            x1 = text_layer(x, y, extended_text_masks, extended_ecg_masks, output_attentions=True)
            y1 = ecg_layer(y, x, extended_ecg_masks, extended_text_masks, output_attentions=True)
            x, y = x1[0], y1[0]
        
        multi_modal_text_feats, multi_modal_ecg_feats = x, y
        multi_modal_text_cls_feats = self.multi_modal_language_pooler(x)
        multi_modal_ecg_cls_feats = self.multi_modal_ecg_pooler(y)
        multi_modal_cls_feats = torch.cat([multi_modal_text_cls_feats, multi_modal_ecg_cls_feats], dim=-1)
        ret.update({
            "multi_modal_text_feats": multi_modal_text_feats,
            "multi_modal_ecg_feats": multi_modal_ecg_feats,
            "multi_modal_cls_feats": multi_modal_cls_feats,
        })

        if features_only:
            return ret

        ret["ecgs"] = ecg
        if mask:
            mlm_logits = self.mlm_head(multi_modal_text_feats)
            if self.mim_layer == -1:
                mim_logits = self.mim_head(multi_modal_ecg_feats, mim_ids_restore)
            else:
                mim_logits = (
                    self.mim_head(ret[f"multi_modal_ecg_feats_{self.mim_layer}"], mim_ids_restore)
                )
            
            itm_logits = self.itm_head(multi_modal_cls_feats)

            ret.update({
                "mlm_logits": mlm_logits,
                "mim_logits": mim_logits,
                "itm_logits": itm_logits
            })
        
        return ret

    def get_logits(self, net_output, **kwargs):
        bsz, tsz, _, _ = net_output["mim_logits"].shape
        res = {
            "mlm_logits": net_output["mlm_logits"].view(-1, self.vocab_size),
            "mim_logits": net_output["mim_logits"].view(bsz, tsz, -1),
            "itm_logits": net_output["itm_logits"]
        }
        
        return res

    def get_targets(self, sample, net_output, norm_pix_loss=True, **kwargs):
        mlm_target = sample["mlm_labels"].view(-1)

        mim_target = sample["net_input"]["ecg"]

        mim_logits = net_output["mim_logits"]
        if mim_target.size(-1) > mim_logits.size(1) * mim_logits.size(-1):
            offset = mim_target.size(-1) - (mim_logits.size(1) * mim_logits.size(-1))
            mim_target = mim_target[:, :, :-offset]

        if norm_pix_loss:
            mean = mim_target.mean(dim=-1, keepdim=True)
            var = mim_target.var(dim=-1, keepdim=True)
            mim_target = (mim_target - mean) / (var + 1.e-6) ** .5
        num_patches = mim_logits.size(1)
        mim_target = mim_target.view(mim_target.size(0), mim_target.size(1), num_patches, -1)
        mim_target = mim_target.permute(0, 2, 1, 3)
        mim_target = mim_target.contiguous().view(mim_target.size(0), mim_target.size(1), -1)

        itm_target = sample["is_aligned"]
        
        return {
            "mlm_target": mlm_target,
            "mim_target": mim_target,
            "itm_target": itm_target.long()
        }

    def extract_features(
        self,
        ecg,
        text,
        ecg_padding_mask,
        text_attention_mask,
        ecg_2,
        ecg_2_padding_mask,
        mask
    ):
        res = self.forward(
            ecg=ecg,
            text=text,
            ecg_padding_mask=ecg_padding_mask,
            text_attention_mask=text_attention_mask,
            ecg_2=ecg_2,
            ecg_2_padding_mask=ecg_2_padding_mask,
            mask=mask,
            features_only=True
        )
        return res

    def remove_pretraining_modules(self):
        self.mlm_head = None
        self.mim_head = None
        self.itm_head = None

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        cfg: M3AEConfig,
        arg_appended=None,
        **kwargs,
    ):
        """
        Load a :class:`~fairseq_signals.models.m3ae.M3AEModel` from a pre-trained model checkpoint.
        
        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (M3AEConfig): cfg to override some arguments of pre-trained model
            arg_appended (dict): dict to be appended to cfg
        """
        
        arg_overrides = {
            "load_pretrained_weights": False,
            "pretrained_model_path": None,
            "drop_rate": cfg.drop_rate
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
        
        args.task.data = cfg.data
        task = tasks.setup_task(args.task, from_checkpoint=True)
        model = task.build_model(args.model)
        
        if hasattr(model, "remove_pretraining_modules"):
            model.remove_pretraining_modules()

        if "ecg_encoder.mask_emb" in state["model"].keys():
            del state["model"]["ecg_encoder.mask_emb"]
        model.load_state_dict(state["model"], strict=True)
        logger.info(f"Loaded pre-trained model parameters from {model_path}")

        return model

@dataclass
class M3AEFinetuningConfig(FinetuningConfig, M3AEConfig):
    pass

class M3AEFinetuningModel(FinetuningModel):
    def __init__(self, cfg: M3AEFinetuningConfig, encoder: M3AEModel):
        super().__init__(cfg, encoder)

        if not cfg.apply_mask and hasattr(self.encoder, "mask_emb"):
            self.encoder.mask_emb = None

        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

    @classmethod
    def build_model(cls, cfg: M3AEFinetuningConfig, task: Task):
        """Build a new model instance."""
        if cfg.model_path and not cfg.no_pretrained_weights:
            encoder = M3AEModel.from_pretrained(cfg.model_path, cfg)
        else:
            encoder = M3AEModel(cfg)
            if hasattr(encoder, "remove_pretraining_modules"):
                encoder.remove_pretraining_modules()

        return cls(cfg, encoder)

    def forward(
        self,
        ecg,
        text,
        ecg_padding_mask=None,
        text_attention_mask=None,
        ecg_2=None,
        ecg_2_padding_mask=None,
        mask=False,
        **kwargs
    ):
        args = {
            "ecg": ecg,
            "text": text,
            "ecg_padding_mask": ecg_padding_mask,
            "text_attention_mask": text_attention_mask,
            "ecg_2": ecg_2,
            "ecg_2_padding_mask": ecg_2_padding_mask,
            "mask": mask
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.encoder.extract_features(**args)

        return res

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

class MIMHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.decoder_hidden_dim = cfg.mim_decoder_hidden_dim
        self.decoder_num_layers = cfg.mim_decoder_num_layers
        self.decoder_num_heads = cfg.mim_decoder_num_heads

        self.decoder_embed = nn.Linear(self.hidden_dim, self.decoder_hidden_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.decoder_pos_embed = PositionalEncoding(self.decoder_hidden_dim, max_len=512)

        self.decoder = Transformer(self.decoder_hidden_dim, self.decoder_num_layers + 1, self.decoder_num_heads)
        self.decoder_norm = LayerNorm(self.decoder_hidden_dim)

        def _conv_out_length(input_length, kernel_size, stride):
            return np.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(cfg.conv_feature_layers)

        dummy_input_length = 5000
        inferred_input_length = dummy_input_length
        for i in range(len(conv_cfg_list)):
            inferred_input_length = _conv_out_length(inferred_input_length, conv_cfg_list[i][1], conv_cfg_list[i][2])
        self.inferred_decoded_size = int(np.floor(dummy_input_length / inferred_input_length))

        self.decoder_pred = nn.Linear(self.decoder_hidden_dim, self.inferred_decoded_size * 12, bias=True)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = self.decoder_pos_embed(x)

        # apply Transformer blocks
        x = x.permute(1, 0, 2)  # (bsz, tsz, csz) -> (tsz, bsz, csz)
        x = self.decoder(x)
        x = x.permute(1, 0, 2)  # (tsz, bsz, csz) -> (bsz, tsz, csz)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        
        # remove cls token
        x = x[:, 1:, :]
        x = x.view(x.size(0), x.size(1), -1, self.inferred_decoded_size)

        return x

class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x): 
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """

        x = x + self.pe[:, :x.size(1)]
        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, x_mask: torch.Tensor):
        if x_mask is not None:
            x_mask = x_mask.to(dtype=torch.bool, device=x.device)
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=x_mask)[0]

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), x_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers - 1)])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for block in self.resblocks:
            x = block(x, x_mask)
        return x