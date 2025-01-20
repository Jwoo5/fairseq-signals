from dataclasses import dataclass, field

from typing import List

import numpy as np
import torch
import torch.nn as nn

from fairseq_signals.utils import utils
from fairseq_signals.data.data_utils import compute_mask_indices
from fairseq_signals.dataclass import ChoiceEnum
from fairseq_signals.models import register_model
from fairseq_signals.models.ecg_language_transformer import (
    ECGLanguageTransformerConfig,
    ECGLanguageTransformerModel
)
from fairseq_signals.modules import (
    GradMultiply,
)

ATTN_MASK_CHOICES = ChoiceEnum(["bidirectional", "bi_ar"])

@dataclass
class MedViLLConfig(ECGLanguageTransformerConfig):
    mask_ratio: List[float] = field(
        default_factory=lambda: [0.15, 0.8, 0.1, 0.1],
        metadata={
            "help": "list of bert masking configurations. "
            "replace ratio, mask ratio, random ratio, and original ratio, respectively"
        }
    )
    attn_mask_type: ATTN_MASK_CHOICES = field(
        default="bi_ar", metadata={"help": "how to choose attention mask type"}
    )

@register_model('medvill', dataclass=MedViLLConfig)
class MedViLLModel(ECGLanguageTransformerModel):
    def __init__(self, cfg: MedViLLConfig):
        super().__init__(cfg)
        self.cfg = cfg
    
        self.mask_ratio = cfg.mask_ratio
        self.attn_mask_type = cfg.attn_mask_type

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.align_proj = nn.Linear(cfg.encoder_embed_dim * 2, 1)
        self.mlm_proj = nn.Linear(cfg.encoder_embed_dim, cfg.vocab_size)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions. """
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
    ):
        B, T, C = x.shape

        replace_ratio = self.mask_ratio[0]
        mask_ratio = self.mask_ratio[1]
        random_token_ratio = self.mask_ratio[2]
        original_ratio = self.mask_ratio[3]

        assert (mask_ratio + random_token_ratio + original_ratio) == 1, (
            mask_ratio, random_token_ratio, original_ratio
        )

        if replace_ratio > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    mask_prob=replace_ratio,
                    mask_length=1,
                    mask_type="static"
                )
                for i in range(B):
                    idc = np.where(mask_indices[i])[0]
                    # replace with [MASK]
                    num_mask = int(
                    # add a random number for probabilistic rounding
                        mask_indices[i].sum() * mask_ratio
                        + np.random.rand()
                    )
                    mask_idc = np.random.choice(
                        idc, size=num_mask, replace=False
                    )
                    x[i][mask_idc] = self.mask_emb

                    
                    # replace with random token
                    rand_prob = random_token_ratio / (random_token_ratio + original_ratio)
                    num_rand = int(
                        (mask_indices[i].sum() - num_mask) * rand_prob
                        + np.random.rand()
                    )
                    if num_rand > 0:
                        rand_embs = self.language_embedding(
                            torch.randint(self.vocab_size, (num_rand,)).to(x.device)
                        )

                        rand_idc = np.random.choice(
                            list(set(idc) - set(mask_idc)), size=num_rand, replace=False
                        )
                        for j, rand_idx in enumerate(rand_idc):
                            x[i][rand_idx] = rand_embs[j]
        else:
            mask_indices = None

        return x, mask_indices

    def forward(
        self,
        ecg,
        text,
        ecg_padding_mask,
        text_padding_mask,
        mask=True,
        features_only=False,
        mask_indices=None,
        **kwargs
    ):
        if self.feature_grad_mult > 0:
            ecg_features = self.feature_extractor(ecg)
            if self.feature_grad_mult != 1.0:
                ecg_features = GradMultiply.apply(ecg_features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                ecg_features = self.feature_extractor(ecg)

        ecg_features = ecg_features.transpose(1,2)
        ecg_features = self.feats_layer_norm(ecg_features)

        if ecg_padding_mask is not None and ecg_padding_mask.any():
            input_lengths = (1 - ecg_padding_mask.long()).sum(-1)
            if input_lengths.dim() > 1:
                for input_len in input_lengths:
                    assert (input_len == input_len[0]).all()
                input_lengths = input_lengths[:,0]
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            ecg_padding_mask = torch.zeros(
                ecg_features.shape[:2], dtype = ecg_features.dtype, device = ecg_features.device
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
        if text_padding_mask is not None and not text_padding_mask.any():
            text_padding_mask = None

        if self.post_extract_proj is not None:
            ecg_features = self.post_extract_proj(ecg_features)
        
        ecg_features_conv = self.conv_pos(ecg_features, channel_first=False)
        ecg_features += ecg_features_conv
        ecg_features_type_embedding = (
            self.token_type_embedding(
                ecg_features.new_zeros(ecg_features.shape[:-1], dtype=int)
            )
        )
        ecg_features += ecg_features_type_embedding

        text_features = self.language_embedding(text)
        text_features = self.layer_norm(text_features)
        text_features = self.dropout_input(text_features)

        if mask:
            bsz = text.size(0)
            special_tokens_mask = text.new_zeros(text.shape).bool()
            for special_token in self.special_tokens:
                if special_token == self.pad_token:
                    continue
                special_tokens_mask |= (text == special_token)
            # select only non-special-tokens in each sample to apply mask only for non-special tokens.
            # each number of special tokens in samples in a batch should be the same.
            # otherwise, this operation will raise RuntimeError.
            masked_text_features = (
                text_features[~special_tokens_mask].view(bsz, -1, self.encoder_embed_dim)
            )
            masked_padding_mask = None
            if text_padding_mask is not None:
                masked_padding_mask = (
                    text_padding_mask[~special_tokens_mask].view(bsz, -1)
                )

            # NOTE num_mask is always the same for each sample, depending on the minimal length.
            # So you may need to check the minimal length of each batch
            #   if avg length of texts is not long enough.
            masked_text_features, masked_indices = self.apply_mask(
                masked_text_features,
                masked_padding_mask,
                mask_indices=mask_indices
            )

            if masked_indices is not None:
                text_features[~special_tokens_mask] = (
                    masked_text_features.view(-1, self.encoder_embed_dim)
                )
                mask_indices = np.zeros(text.shape, dtype=bool)
                mask_indices[~special_tokens_mask.cpu()] = (masked_indices.flatten())

                # TODO text[mask_indices] sometimes raise IndexError
                # need to investigate why this happens
                y = text.view(-1)[mask_indices.flatten()].view(text.size(0), -1)
            else:
                y = text
        else:
            y = text
            mask_indices = None

        text_features_pos = self.position_embedding(
            torch.arange(
                text_features.size(1),
                device=text_features.device
            ).repeat((text_features.size(0), 1))
        )
        text_features += text_features_pos
        text_features_type_embedding = (
            self.token_type_embedding(
                text_features.new_ones(text_features.shape[:-1], dtype=int)
            )
        )
        text_features += text_features_type_embedding

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

        if self.attn_mask_type == "bi_ar":
            attn_mask = x.new_zeros((x.size(1), x.size(1))).bool()
            attn_mask[ecg_features.size(1):, ecg_features.size(1):] = (
                (
                    1 - torch.tril(
                        torch.full((text_features.size(1), text_features.size(1)), fill_value=1)
                    )
                ).bool()
            )
        else:
            attn_mask = None
        x_result = self.encoder(x, padding_mask=padding_mask, attn_mask=attn_mask)
        features = x_result["x"]

        ecg_features = features[:, :ecg_features.size(1)]
        text_features = features[:, ecg_features.size(1):]

        # TODO text[mask_indices] sometimes raise IndexError
        # same with the above reports
        mlm_x = (
            text_features.contiguous().view(-1, text_features.size(-1))[mask_indices.flatten()]
        )
        mlm_x = mlm_x.view(text_features.size(0), -1, text_features.size(-1))
        mlm_x = self.mlm_proj(mlm_x)
        mlm_x = mlm_x.view(-1, mlm_x.size(-1))
        mlm_y = y.view(-1)

        if padding_mask is not None and padding_mask.any():
            features[padding_mask] = 0

        if ecg_padding_mask is not None and ecg_padding_mask.any():
            ecg_features[ecg_padding_mask] = 0
        ecg_feats = torch.div(ecg_features.sum(dim=1), (ecg_features != 0).sum(dim=1))
        if text_padding_mask is not None and text_padding_mask.any():
            text_features[text_padding_mask] = 0
        text_feats = torch.div(text_features.sum(dim=1), (text_features != 0).sum(dim=1))

        align_x = self.align_proj(
            torch.cat([ecg_feats, text_feats], dim=1)
        )

        result = {
            "mlm_x": mlm_x,
            "mlm_y": mlm_y,
            "align_x": align_x,
        }

        return result
    
    def extract_features(
        self,
        ecg,
        ecg_padding_mask,
        text,
        text_padding_mask,
        ecg_2,
        ecg_2_padding_mask,
    ):
        res = super().forward(
            ecg=ecg,
            text=text,
            ecg_padding_mask=ecg_padding_mask,
            text_padding_mask=text_padding_mask,
            ecg_2=ecg_2,
            ecg_2_padding_mask=ecg_2_padding_mask
        )
        return res
    
    def get_logits(self, net_output, **kwargs):
        res = {
            "align_x": torch.sigmoid(net_output["align_x"].squeeze(-1)),
            "mlm_x": net_output["mlm_x"]
        }
        return res

    def get_targets(self, sample, net_output, **kwargs):
        align_y = sample["is_aligned"]
        mlm_y = net_output["mlm_y"]
        res = {
            "align_y": align_y,
            "mlm_y": mlm_y
        }
        return res
    
    def remove_pretraining_modules(self):
        self.align_proj = None
        self.mlm_proj = None