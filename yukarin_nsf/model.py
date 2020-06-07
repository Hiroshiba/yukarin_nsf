from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from pytorch_trainer import report
from torch import Tensor, nn
from torch.nn import functional as F

from yukarin_nsf.config import ModelConfig, NetworkConfig
from yukarin_nsf.network.discriminator import Discriminator, create_discriminator
from yukarin_nsf.network.predictor import Predictor, create_predictor


@dataclass
class Networks:
    predictor: Predictor
    discriminator: Optional[Discriminator]


def create_network(config: NetworkConfig):
    return Networks(
        predictor=create_predictor(config),
        discriminator=create_discriminator(config)
        if config.discriminator_type is not None
        else None,
    )


class DiscriminatorInputType(str, Enum):
    gan = "gan"
    cgan = "cgan"


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
        size=(
            silence.shape[0],
            (silence.shape[1] - fft_size) // hop_length + 1,
            window_length,
        ),
        stride=(1, hop_length, 1),
    )
    return ~(silence.all(dim=2))


def stft_weight(
    signal: Tensor, fft_size: int, hop_length: int, window_length: int, threshold=1e-6
):
    x = stft(
        signal, fft_size=fft_size, hop_length=hop_length, window_length=window_length,
    )

    real, image = x[..., 0], x[..., 1]
    amplitude = real ** 2 + image ** 2

    m = amplitude.max(1, keepdim=True)[0].expand_as(amplitude)
    f = m > threshold

    weight = torch.zeros_like(real)
    weight[f] = amplitude[f] / m[f]
    return weight


def amplitude_distance(
    x: Tensor, t: Tensor, mask: Tensor = None, weight: Tensor = None, epsilon=1e-6
):
    if mask is not None:
        x = x.transpose(1, 2)[mask]
        t = t.transpose(1, 2)[mask]
        if weight is not None:
            weight = weight.transpose(1, 2)[mask]

    if weight is not None:
        weight = weight.unsqueeze(-1).expand_as(x)
        x *= weight
        t *= weight

    x_real, x_image = x[..., 0], x[..., 1]
    t_real, t_image = t[..., 0], t[..., 1]

    x_amplitude = x_real ** 2 + x_image ** 2 + epsilon
    t_amplitude = t_real ** 2 + t_image ** 2 + epsilon
    return torch.mean((torch.log(x_amplitude) - torch.log(t_amplitude)) ** 2) / 2


class Model(nn.Module):
    def __init__(
        self, model_config: ModelConfig, networks: Networks, local_padding_length: int,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.predictor = networks.predictor
        self.discriminator = networks.discriminator
        self.local_padding_length = local_padding_length

    def forward(
        self,
        wave: Tensor,
        silence: Tensor,
        local: Tensor,
        source: Tensor,
        source2: Tensor,
        signal: Tensor,
        speaker_id: Optional[Tensor] = None,
    ):
        assert silence.is_contiguous()

        batch_size = wave.shape[0]
        values = {}

        output = self.predictor(
            source=source,
            local=local,
            local_padding_length=self.local_padding_length,
            speaker_id=speaker_id,
        )

        # stft
        stft_loss_list = [
            amplitude_distance(
                x=stft(
                    output,
                    fft_size=stft_config["fft_size"],
                    hop_length=stft_config["hop_length"],
                    window_length=stft_config["window_length"],
                ),
                t=stft(
                    wave,
                    fft_size=stft_config["fft_size"],
                    hop_length=stft_config["hop_length"],
                    window_length=stft_config["window_length"],
                ),
                mask=stft_mask(
                    silence=silence,
                    fft_size=stft_config["fft_size"],
                    hop_length=stft_config["hop_length"],
                    window_length=stft_config["window_length"],
                )
                if self.model_config.eliminate_silence
                else None,
                weight=stft_weight(
                    signal=signal,
                    fft_size=stft_config["fft_size"],
                    hop_length=stft_config["hop_length"],
                    window_length=stft_config["window_length"],
                )
                if self.model_config.use_stft_weight
                else None,
            )
            for stft_config in self.model_config.stft_config
        ]
        stft_loss = torch.mean(torch.stack(stft_loss_list))
        loss = stft_loss
        values["loss_stft"] = stft_loss

        # adversarial
        if self.model_config.discriminator_input_type is not None:
            if self.model_config.discriminator_input_type == DiscriminatorInputType.gan:
                c = None
            else:
                c = source
            fake = self.discriminator(x=output, c=c)

            mask = self.discriminator.generate_mask(silence=silence)
            adv_loss = (
                F.softplus(-fake)[mask].mean()
                * self.model_config.adversarial_loss_scale
            )
            loss += adv_loss
            values["loss_adv"] = adv_loss

        # report
        values["loss"] = loss

        for l, stft_config in zip(stft_loss_list, self.model_config.stft_config):
            key = "loss_{fft_size}_{hop_length}_{window_length}".format(**stft_config)
            values[key] = l

        if not self.training:
            values = {key: (l, batch_size) for key, l in values.items()}  # add weight
        report(values, self)

        return loss


class DiscriminatorModel(nn.Module):
    def __init__(
        self, model_config: ModelConfig, networks: Networks, local_padding_length: int,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.predictor = networks.predictor
        self.discriminator = networks.discriminator
        self.local_padding_length = local_padding_length

    def forward(
        self,
        wave: Tensor,
        silence: Tensor,
        local: Tensor,
        source: Tensor,
        source2: Tensor,
        signal: Tensor,
        speaker_id: Optional[Tensor] = None,
    ):
        assert silence.is_contiguous()

        batch_size = wave.shape[0]
        values = {}

        with torch.no_grad():
            output = self.predictor(
                source=source,
                local=local,
                local_padding_length=self.local_padding_length,
                speaker_id=speaker_id,
            )

        # adversarial
        mask = self.discriminator.generate_mask(silence=silence)

        if self.model_config.discriminator_input_type == DiscriminatorInputType.gan:
            c = None
        else:
            c = source2

        fake = self.discriminator(x=output, c=c)[mask]
        fake_loss = F.softplus(fake).mean()
        values["loss_fake"] = fake_loss
        values["recall_fake"] = (fake < 0).float().mean()

        real = self.discriminator(x=wave, c=c)[mask]
        real_loss = F.softplus(-real).mean()
        values["loss_real"] = real_loss
        values["recall_real"] = (real > 0).float().mean()

        # report
        loss = fake_loss + real_loss
        values["loss"] = loss

        if not self.training:
            values = {key: (l, batch_size) for key, l in values.items()}  # add weight
        report(values, self)

        return loss
