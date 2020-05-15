from enum import Enum

import numpy
import torch
from torch import Tensor, nn
from typing import Optional

from yukarin_nsf.config import NetworkConfig


class DiscriminatorType(str, Enum):
    wavegan = 'wavegan'
    cgan = 'cgan'


class Discriminator(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            layer_num: int,
    ):
        super().__init__()
        self.layer_num = layer_num

        self.head = nn.Sequential(*[
            nn.utils.weight_norm(nn.Conv1d(
                in_channels=input_size,
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
            c: Optional[Tensor],
    ):
        """
        :param x: float (batch_size, length)
        :param c: float (batch_size, length)
        :return:
            output: float (batch_size, ?)
        """
        if c is None:
            x = x.unsqueeze(1)
        else:
            x = torch.stack([x, c], dim=1)

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


def create_discriminator(config: NetworkConfig):
    if config.discriminator_type == DiscriminatorType.wavegan:
        input_size = 1
    elif config.discriminator_type == DiscriminatorType.cgan:
        input_size = 2
    else:
        raise ValueError(config.discriminator_type)

    return Discriminator(
        input_size=input_size,
        hidden_size=config.discriminator_hidden_size,
        layer_num=config.discriminator_layer_num,
    )
