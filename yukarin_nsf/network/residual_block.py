import math
from typing import Optional, Callable

import torch
from torch import nn, Tensor


class ResidualBlock(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            condition_size: Optional[int],
            dilation: int,
            is_last: bool,
            no_padding=False,
            output_activation_function: Callable[[Tensor], Tensor] = None,
    ):
        super().__init__()
        self.dilation = dilation
        self.no_padding = no_padding
        self.output_activation_function = output_activation_function

        self.input_conv = nn.utils.weight_norm(nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size * 2,
            kernel_size=3,
            dilation=dilation,
            padding=dilation if not no_padding else 0,
        ))

        self.condition_conv = nn.utils.weight_norm(nn.Conv1d(
            in_channels=condition_size,
            out_channels=hidden_size * 2,
            kernel_size=1,
            bias=False,
        )) if condition_size is not None else None

        self.skip_conv = nn.utils.weight_norm(nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=1,
        ))

        if not is_last:
            self.output_conv = nn.utils.weight_norm(nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=1,
            ))
        else:
            self.output_conv = None

    def forward(
            self,
            x: Tensor,
            c: Optional[Tensor],
    ):
        assert (c is None) == (self.condition_conv is None)

        h = self.input_conv(x)
        if c is not None:
            h += self.condition_conv(c)

        a, b = h.split(h.shape[1] // 2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(b)

        if self.output_conv is not None:
            output = self.output_conv(h)
            if self.output_activation_function is not None:
                output = self.output_activation_function(output)
            if self.no_padding:
                x = x[:, :, self.dilation:-self.dilation]
            output = (output + x) * math.sqrt(0.5)
        else:
            output = None

        skip = self.skip_conv(h)
        return output, skip
