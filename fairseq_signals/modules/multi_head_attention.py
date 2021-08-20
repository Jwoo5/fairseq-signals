# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head, dropout):
        super().__init__()
        self.d_head = d_head
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / (self.d_head ** 0.5)
    
    def forward(self, Q, K, V, padding_mask=None):
        scores = torch.matmul(Q, K.transpose(-1,-2)).mul_(self.scale)
        if padding_mask is not None and padding_mask.any():
            scores = scores.masked_fill(padding_mask, float("-inf"))
        attn_prob = F.softmax(scores, dim = -1)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, V)

        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_heads
        self.d_head = embed_dim // n_heads

        self.W_Q = nn.Linear(self.embed_dim, self.n_head * self.d_head)
        self.W_K = nn.Linear(self.embed_dim, self.n_head * self.d_head)
        self.W_V = nn.Linear(self.embed_dim, self.n_head * self.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head, dropout)
        self.linear = nn.Linear(self.n_head * self.d_head, self.embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, padding_mask=None):
        # (batch_size, n_seq, d_model)
        batch_size = Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)

        context = self.scaled_dot_attn(q_s, k_s, v_s, padding_mask)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.n_head * self.d_head)

        output = self.linear(context)
        output = self.dropout(output)

        return output