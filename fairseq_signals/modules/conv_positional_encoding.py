import math

import torch.nn as nn

from fairseq_signals.modules import (
    SamePad,
)

class ConvPositionalEncoding(nn.Module):
    def __init__(self, args):
        super().__init__()

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

        self.pos_conv = nn.utils.parametrizations.weight_norm(self.pos_conv, name = "weight", dim = 2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

    def forward(self, x, channel_first=False):
        if not channel_first:
            x = x.transpose(1,2)
        x_conv = self.pos_conv(x).transpose(1,2)

        return x_conv