import math

import torch
from torch import nn, Tensor


class ResidualBlock(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            condition_size: int,
            dilation: int,
            is_last: bool,
    ):
        super().__init__()

        self.input_conv = nn.utils.weight_norm(nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size * 2,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        ))

        self.condition_conv = nn.utils.weight_norm(nn.Conv1d(
            in_channels=condition_size,
            out_channels=hidden_size * 2,
            kernel_size=1,
            bias=False,
        ))

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
            c: Tensor,
    ):
        h = self.input_conv(x) + self.condition_conv(c)

        a, b = h.split(h.shape[1] // 2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(b)

        if self.output_conv is not None:
            output = (self.output_conv(h) + x) * math.sqrt(0.5)
        else:
            output = None

        skip = self.skip_conv(h)
        return output, skip
