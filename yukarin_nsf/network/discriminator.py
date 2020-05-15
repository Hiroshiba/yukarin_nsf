import math
from enum import Enum

import numpy
from torch import Tensor, nn

from yukarin_nsf.network.residual_block import ResidualBlock


class DiscriminatorType(str, Enum):
    wavegan = 'wavegan'


class Discriminator(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            layer_num: int,
    ):
        super().__init__()
        self.layer_num = layer_num

        self.head = nn.Sequential(*[
            nn.utils.weight_norm(nn.Conv1d(
                in_channels=1,
                out_channels=hidden_size,
                kernel_size=1,
            )),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        ])

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                hidden_size=hidden_size,
                condition_size=None,
                dilation=2 ** i_layer,
                is_last=i_layer == layer_num - 1,
                no_padding=True,
                output_activation_function=nn.LeakyReLU(inplace=True, negative_slope=0.2),
            )
            for i_layer in range(layer_num)
        ])

        self.tail = nn.utils.weight_norm(nn.Conv1d(
            in_channels=hidden_size,
            out_channels=1,
            kernel_size=1,
        ))

    def forward(
            self,
            x: Tensor,
    ):
        """
        :param x: float (batch_size, length)
        :return:
            output: float (batch_size, ?)
        """
        x = x.unsqueeze(1)
        x = self.head(x)

        output = None
        for i_layer, residual_block in enumerate(self.residual_blocks):
            x, s = residual_block(x=x, c=None)

            if i_layer < self.layer_num - 1:
                pad = numpy.sum(2 ** numpy.arange(i_layer + 1, self.layer_num))
                s = s[:, :, pad:-pad]

            if output is None:
                output = s
            else:
                output += s

        output *= math.sqrt(1.0 / len(self.residual_blocks))

        output = self.tail(output)
        output = output.squeeze(1)
        return output

    def generate_mask(
            self,
            silence: Tensor,
    ):
        """
        :param silence: bool (batch_size, length)
        :return:
            output: bool (batch_size, ?)
        """
        window_length = 1 + numpy.sum(2 ** numpy.arange(1, self.layer_num + 1))

        silence = silence.unsqueeze(2)
        silence = silence.as_strided(
            size=(silence.shape[0], silence.shape[1] - (window_length - 1), window_length),
            stride=(1, 1, 1),
        )
        return ~(silence.all(dim=2))
