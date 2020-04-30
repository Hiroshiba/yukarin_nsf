from dataclasses import dataclass
from typing import Optional
from warnings import warn

import numpy
import torch
from pytorch_trainer import report
from torch import nn, Tensor

from yukarin_nsf.config import ModelConfig, NetworkConfig
from yukarin_nsf.network.predictor import create_predictor, Predictor


@dataclass
class Networks:
    predictor: Predictor


def create_network(config: NetworkConfig):
    return Networks(
        predictor=create_predictor(config),
    )


def stft(x: Tensor, fft_size: int, hop_length: int, window_length: int):
    return torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=window_length,
        window=torch.hann_window(window_length),
        center=False,
    )


def stft_mask(silence: Tensor, fft_size: int, hop_length: int, window_length: int):
    silence = silence.unsqueeze(2)
    silence = silence.as_strided(
        size=(silence.shape[0], (silence.shape[1] - fft_size) // hop_length + 1, window_length),
        stride=(1, hop_length, 1),
    )
    return ~(silence.all(dim=2))


def amplitude_distance(x: Tensor, t: Tensor, mask: Tensor = None, epsilon=1e-6):
    if mask is not None:
        assert torch.any(mask)

        mask = mask.reshape(mask.shape[0], 1, mask.shape[1], 1).expand_as(t)
        x = x[mask]
        t = t[mask]

    x_real, x_image = x[..., 0], x[..., 1]
    t_real, t_image = t[..., 0], t[..., 1]

    x_amplitude = x_real ** 2 + x_image ** 2 + epsilon
    t_amplitude = t_real ** 2 + t_image ** 2 + epsilon
    return torch.mean((torch.log(x_amplitude) - torch.log(t_amplitude)) ** 2)


class Model(nn.Module):
    def __init__(
            self,
            model_config: ModelConfig,
            networks: Networks,
            local_padding_length: int,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.predictor = networks.predictor
        self.local_padding_length = local_padding_length

    def __call__(
            self,
            wave: Tensor,
            silence: Tensor,
            local: Tensor,
            source: Tensor,
            speaker_id: Optional[Tensor] = None,
    ):
        assert silence.is_contiguous()

        batch_size = wave.shape[0]

        output = self.predictor(
            source=source,
            local=local,
            local_padding_length=self.local_padding_length,
            speaker_id=speaker_id,
        )

        loss_list = [
            amplitude_distance(
                x=stft(
                    output,
                    fft_size=stft_config['fft_size'],
                    hop_length=stft_config['hop_length'],
                    window_length=stft_config['window_length'],
                ),
                t=stft(
                    wave,
                    fft_size=stft_config['fft_size'],
                    hop_length=stft_config['hop_length'],
                    window_length=stft_config['window_length'],
                ),
                mask=stft_mask(
                    silence=silence,
                    fft_size=stft_config['fft_size'],
                    hop_length=stft_config['hop_length'],
                    window_length=stft_config['window_length'],
                ) if self.model_config.eliminate_silence else None
            )
            for stft_config in self.model_config.stft_config
        ]
        loss = torch.sum(torch.stack(loss_list))

        # report
        values = dict(
            loss=loss,
        )
        for l, stft_config in zip(loss_list, self.model_config.stft_config):
            key = 'loss_{fft_size}_{hop_length}_{window_length}'.format(**stft_config)
            values[key] = l

        if not self.training:
            values = {key: (l, batch_size) for key, l in values.items()}  # add weight
        report(values, self)

        return loss
