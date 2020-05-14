from pathlib import Path
from pprint import pprint
from typing import Callable, Dict, Optional

import numpy
import torch
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave
from pytorch_trainer import Reporter
from pytorch_trainer.iterators import SerialIterator
from torch.optim import Adam
from torch.utils.data import Dataset

from yukarin_nsf.dataset import BaseWaveDataset
from yukarin_nsf.model import Model, DiscriminatorModel
from yukarin_nsf.updater import Updater
from yukarin_nsf.utility.dataset_utility import default_convert


def get_data_directory() -> Path:
    return Path(__file__).parent.relative_to(Path.cwd()) / 'data'


class SignWaveDataset(BaseWaveDataset):
    def __init__(
            self,
            sampling_length: int,
            sampling_rate: int,
            local_padding_length: int,
            local_scale: int,
            f0_index: int = 0,
            frequency: float = 440
    ) -> None:
        super().__init__(
            sampling_length=sampling_length,
            local_padding_length=local_padding_length,
            min_not_silence_length=0,
            f0_index=f0_index,
        )
        self.sampling_rate = sampling_rate
        self.local_scale = local_scale
        self.frequency = frequency

    def __len__(self):
        return 100

    def __getitem__(self, i):
        sampling_rate = self.sampling_rate
        length = self.sampling_length
        rand = numpy.random.rand()

        wave = (numpy.arange(length, dtype=numpy.float32) * self.frequency / sampling_rate + rand) * 2 * numpy.pi
        wave = numpy.sin(wave) / 2

        local = numpy.ones(shape=(length // self.local_scale, 1), dtype=numpy.float32)
        local = numpy.log(local * self.frequency)

        silence = numpy.zeros(shape=(length,), dtype=numpy.bool)

        return default_convert(self.make_input(
            wave_data=Wave(wave=wave, sampling_rate=sampling_rate),
            silence_data=SamplingData(array=silence, rate=sampling_rate),
            local_data=SamplingData(array=local, rate=sampling_rate // self.local_scale),
        ))


def train_support(
        batch_size: int,
        use_gpu: bool,
        model: Model,
        discriminator_model: Optional[DiscriminatorModel],
        dataset: Dataset,
        iteration: int,
        first_hook: Callable[[Dict], None] = None,
        last_hook: Callable[[Dict], None] = None,
):
    optimizer = Adam(model.parameters(), lr=0.001)
    if discriminator_model is not None:
        discriminator_optimizer = Adam(discriminator_model.parameters(), lr=0.001)
    else:
        discriminator_optimizer = None

    train_iter = SerialIterator(dataset, batch_size)

    if use_gpu:
        device = torch.device('cuda')
        model.to(device)
        if discriminator_model is not None:
            discriminator_model.to(device)
    else:
        device = torch.device('cpu')

    updater = Updater(
        iterator=train_iter,
        optimizer=optimizer,
        discriminator_optimizer=discriminator_optimizer,
        model=model,
        discriminator_model=discriminator_model,
        device=device,
    )

    reporter = Reporter()
    reporter.add_observer('main', model)
    if discriminator_model is not None:
        reporter.add_observer('discriminator', discriminator_model)

    observation: Dict = {}
    for i in range(iteration):
        with reporter.scope(observation):
            updater.update()

        if i % 100 == 0:
            print('iteration', i)
            pprint(observation)

        if i == 0:
            if first_hook is not None:
                first_hook(observation)

    pprint(observation)
    if last_hook is not None:
        last_hook(observation)
