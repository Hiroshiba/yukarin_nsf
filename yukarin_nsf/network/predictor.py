from enum import Enum
from typing import Optional

import torch
from torch import nn, Tensor

from yukarin_nsf.config import NetworkConfig
from yukarin_nsf.network.wavenet import WaveNet


class NeuralFilterType(str, Enum):
    gru = 'gru'
    wavenet = 'wavenet'


class Predictor(nn.Module):
    def __init__(
            self,
            speaker_size: int,
            speaker_embedding_size: int,
            local_size: int,
            local_scale: int,
            local_layer_num: int,
            condition_size: int,
            neural_filter_type: NeuralFilterType,
            neural_filter_layer_num: int,
            neural_filter_stack_num: Optional[int],
            neural_filter_hidden_size: int,
    ):
        super().__init__()
        self.speaker_size = speaker_size
        self.local_size = local_size
        self.local_scale = local_scale
        self.neural_filter_type = neural_filter_type

        if self.with_speaker:
            self.speaker_embedder = nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )
        else:
            self.speaker_embedder = None

        self.local_encoder = nn.GRU(
            input_size=local_size + (speaker_embedding_size if self.with_speaker else 0),
            hidden_size=condition_size,
            num_layers=local_layer_num,
            batch_first=True,
            bidirectional=True,
        )

        if neural_filter_type == NeuralFilterType.gru:
            self.neural_filter = nn.GRU(
                input_size=1 + 2 * condition_size,
                hidden_size=neural_filter_hidden_size,
                num_layers=neural_filter_layer_num,
                batch_first=True,
            )
            self.neural_filter_cap = nn.Linear(
                in_features=neural_filter_hidden_size,
                out_features=1,
            )
        elif neural_filter_type == NeuralFilterType.wavenet:
            self.neural_filter = WaveNet(
                input_size=1,
                hidden_size=neural_filter_hidden_size,
                condition_size=1 + 2 * condition_size,
                stack_num=neural_filter_stack_num,
                layer_num_per_stack=neural_filter_layer_num,
            )
            self.neural_filter_cap = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(
                    in_features=neural_filter_hidden_size,
                    out_features=neural_filter_hidden_size,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(
                    in_features=neural_filter_hidden_size,
                    out_features=1,
                ),
            )
        else:
            raise ValueError(neural_filter_type)

    @property
    def with_speaker(self):
        return self.speaker_size > 0

    def forward(
            self,
            source: Tensor,
            local: Tensor,
            local_padding_length: int,
            speaker_id: Tensor = None,
    ):
        """
        :param source: float (batch_size, wave_length)
        :param local: float (batch_size, local_length, ?)
        :param speaker_id: int (batch_size, )
        :return:
            output: float (batch_size, wave_length)
        """
        assert local.shape[2] == self.local_size, f'{local.shape[2]} {self.local_size}'

        batch_size = local.shape[0]
        local_length = local.shape[1]

        source = source.unsqueeze(2)

        if self.with_speaker:
            speaker = self.speaker_embedder(speaker_id)
            speaker = speaker.unsqueeze(1)
            speaker = speaker.expand(batch_size, local_length, speaker.shape[2])
            local = torch.cat((local, speaker), dim=2)

        condition, _ = self.local_encoder(local)

        condition = condition.unsqueeze(2)
        condition = condition.expand(batch_size, local_length, self.local_scale, condition.shape[3])
        condition = condition.reshape(batch_size, local_length * self.local_scale, condition.shape[3])

        if local_padding_length > 0:
            condition = condition[:, local_padding_length:-local_padding_length]

        if self.neural_filter_type == NeuralFilterType.gru:
            output = torch.cat((source, condition), dim=2)
            output, _ = self.neural_filter(output)
        elif self.neural_filter_type == NeuralFilterType.wavenet:
            condition = torch.cat((source, condition), dim=2)
            output = self.neural_filter(x=source, c=condition)
        else:
            raise ValueError(self.neural_filter_type)

        output = self.neural_filter_cap(output).squeeze(2)
        return output


def create_predictor(config: NetworkConfig):
    return Predictor(
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        local_size=config.local_size,
        local_scale=config.local_scale,
        local_layer_num=config.local_layer_num,
        condition_size=config.condition_size,
        neural_filter_type=NeuralFilterType(config.neural_filter_type),
        neural_filter_layer_num=config.neural_filter_layer_num,
        neural_filter_stack_num=config.neural_filter_stack_num,
        neural_filter_hidden_size=config.neural_filter_hidden_size,
    )
