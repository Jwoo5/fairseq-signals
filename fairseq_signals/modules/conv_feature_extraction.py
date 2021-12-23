from typing import List, Tuple
import torch.nn as nn

from fairseq_signals.modules import (
    TransposeLast,
    Fp32LayerNorm,
    Fp32GroupNorm
)

class ConvFeatureExtraction(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        in_d: int = 1,
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm = False,
            is_group_norm = False,
            conv_bias = False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride = stride, bias = conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, dim, affine = True),
                        TransposeLast()
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm = mode == "layer_norm",
                    is_group_norm = mode == "default" and i == 0,
                    conv_bias = conv_bias,
                )
            )
            in_d = dim
    
    def forward(self, x):
        # B x T -> B x C x T
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x

class TransposedConvFeatureExtraction(nn.Module):
    def __init__(
        self,
        conv_transposed_layers: List[Tuple[int, int, int]],
        in_d: int=768,
        dropout: float=0.0,
        mode: str="default",
        conv_bias: bool=True
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
            is_last=False,
        ):
            def make_conv_transposed():
                transposed_conv = nn.ConvTranspose1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(transposed_conv.weight)
                return transposed_conv
            
            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                b = [
                    make_conv_transposed(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, dim, affine=True),
                        TransposeLast()
                    ),
                ]
            elif is_group_norm:
                b = [
                    make_conv_transposed(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                ]
            else:
                b = [make_conv_transposed(), nn.Dropout(p=dropout),]

            if not is_last:
                b.append(nn.GELU())
            return nn.Sequential(*b)
        
        self.conv_transposed_layers = nn.ModuleList()
        for i, cl in enumerate(conv_transposed_layers):
            assert len(cl) == 3, "invalid transposed conv definition " + str(cl)
            (dim, k, stride) = cl

            self.conv_transposed_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm= mode == "layer_norm",
                    is_group_norm= mode == "default" and i == 0,
                    conv_bias=conv_bias,
                    is_last = i+1 == len(conv_transposed_layers)
                )
            )
            in_d = dim
    
    def forward(self, x):
        # B x T -> B x C x T
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
        
        for transposed_conv in self.conv_transposed_layers:
            x = transposed_conv(x)
        
        return x