import torch.nn as nn
import torch.nn.functional as F
from fairseq_signals.modules import MultiHeadAttention

class PositionwiseFeedForwardLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_dim)
        self.fc2 = nn.Linear(self.ffn_dim, self.embed_dim)
        self.active = F.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        output = self.fc1(inputs)
        output = self.active(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_heads,
        ffn_dim,
        dropout,
        attention_dropout,
        activation_dropout
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_heads
        self.d_head = embed_dim // n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout

        self.self_attn = MultiHeadAttention(self.embed_dim, self.n_head, self.attention_dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm_layer1 = nn.LayerNorm(self.embed_dim, eps = 1e-6)
        self.pos_ffn = PositionwiseFeedForwardLayer(self.embed_dim, self.ffn_dim, activation_dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_layer2 = nn.LayerNorm(self.embed_dim, eps = 1e-6)
    
    def forward(self, inputs, padding_mask=None):
        att_outputs = self.self_attn(inputs, inputs, inputs, padding_mask=padding_mask)
        att_outputs = self.dropout1(att_outputs)
        att_outputs = self.norm_layer1(att_outputs + inputs)

        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.norm_layer2(ffn_outputs + att_outputs)

        return ffn_outputs

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        n_layer,
        embed_dim,
        n_heads,
        ffn_dim,
        dropout,
        attention_dropout,
        activation_dropout
    ):
        super().__init__()
        self.n_layer = n_layer
        self.embed_dim = embed_dim
        self.n_head = n_heads
        self.d_head = embed_dim // n_heads
        self.ffn_dim = ffn_dim
        
        self.layers = nn.ModuleList(
            [EncoderLayer(
                    self.embed_dim,
                    self.n_head,
                    self.ffn_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout
                )
                for _ in range(self.n_layer)]
        )

    def forward(self, inputs, padding_mask=None):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, padding_mask=padding_mask)
        return outputs