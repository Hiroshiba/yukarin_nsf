from enum import Enum

import numpy
from torch import Tensor, nn


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

        convs = []
        for i_layer in range(layer_num):
            convs.append(nn.utils.weight_norm(nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=3,
                dilation=2 ** i_layer,
                padding=0,
            )))
            convs.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.convs = nn.Sequential(*convs)

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
        x = self.convs(x)
        x = self.tail(x)
        x = x.squeeze(1)
        return x

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
