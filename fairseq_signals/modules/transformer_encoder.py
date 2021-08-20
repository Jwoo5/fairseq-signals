# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn as nn
import torch.nn.functional as F

from fairseq_signals.modules import (
    TransformerEncoderLayer,
    MultiHeadAttention,
    SamePad,
    LayerNorm
)

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiHeadAttention):
        module.W_Q.weight.data.normal_(mean=0.0, std=0.02)
        module.W_K.weight.data.normal_(mean=0.0, std=0.02)
        module.W_V.weight.data.normal_(mean=0.0, std=0.02)

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        
        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size = args.conv_pos,
            padding = args.conv_pos // 2,
            groups = args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean = 0, std = std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name = "weight", dim = 2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.encoder = TransformerEncoderLayer(
            n_layer = args.encoder_layers,
            embed_dim = self.embedding_dim,
            n_heads = args.encoder_attention_heads,
            ffn_dim = args.encoder_ffn_embed_dim,
            dropout = self.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout            
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)

        # self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)
    
    def extract_features(self, x, padding_mask = None):
        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0
            padding_mask = padding_mask.transpose(-1,-2).unsqueeze(1).unsqueeze(2)
        x_conv = self.pos_conv(x.transpose(1,2))
        x_conv = x_conv.transpose(1,2)

        x += x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training = self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        x = self.encoder(x, padding_mask=padding_mask)
        # T x B x C -> B x T x C
        x = x.transpose(0,1)

        return x
    
    def forward(self, x, padding_mask = None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)
        return x