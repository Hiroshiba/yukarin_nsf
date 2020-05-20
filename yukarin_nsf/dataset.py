import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict

import numpy
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave
from torch.utils.data import Dataset, ConcatDataset

from yukarin_nsf.config import DatasetConfig
from yukarin_nsf.utility.dataset_utility import default_convert


@dataclass
class Input:
    wave: Wave
    silence: SamplingData
    local: SamplingData


@dataclass
class LazyInput:
    path_wave: Path
    path_silence: Path
    path_local: Path

    def generate(self):
        return Input(
            wave=Wave.load(self.path_wave),
            silence=SamplingData.load(self.path_silence),
            local=SamplingData.load(self.path_local),
        )


def generate_source(log_f0: numpy.ndarray, local_rate: int, sampling_rate: int):
    f0 = numpy.exp(log_f0)
    f0[log_f0 == 0] = 0

    f0 = numpy.repeat(f0, sampling_rate // local_rate)
    voiced = f0 != 0

    source = numpy.empty(len(f0), dtype=numpy.float32)

    points = numpy.where(voiced[1:] != voiced[:-1])[0] + 1
    for start, end in zip(numpy.r_[0, points], numpy.r_[points, len(f0)]):
        if voiced[start]:
            r = numpy.random.uniform(-numpy.pi, numpy.pi)
            source[start:end] = numpy.sin(2 * numpy.pi * numpy.cumsum(f0[start:end]) / sampling_rate + r) * 0.1
            source[start:end] += numpy.random.randn(end - start) * 0.003
        else:
            source[start:end] = numpy.random.randn(end - start) / 3 * 0.1
    return source


class BaseWaveDataset(Dataset):
    def __init__(
            self,
            sampling_length: int,
            local_padding_length: int,
            min_not_silence_length: int,
            f0_index: int,
            only_noise_source: bool,
    ) -> None:
        self.sampling_length = sampling_length
        self.local_padding_length = local_padding_length
        self.min_not_silence_length = min_not_silence_length
        self.f0_index = f0_index
        self.only_noise_source = only_noise_source

    @staticmethod
    def extract_input(
            sampling_length: int,
            wave_data: Wave,
            silence_data: SamplingData,
            local_data: SamplingData,
            local_padding_length: int,
            min_not_silence_length: int,
            f0_index: int,
            only_noise_source: bool,
            padding_value=0,
    ):
        """
        :return:
            wave: (sampling_length, )
            silence: (sampling_length, )
            local: (sampling_length // scale + pad, )
        """
        sr = wave_data.sampling_rate
        sl = sampling_length

        assert sr % local_data.rate == 0
        l_scale = int(sr // local_data.rate)

        length = len(local_data.array) * l_scale
        assert abs(length - len(wave_data.wave)) < l_scale * 4, f'{abs(length - len(wave_data.wave))} {l_scale}'

        assert local_padding_length % l_scale == 0
        l_pad = local_padding_length // l_scale

        l_length = length // l_scale
        l_sl = sl // l_scale

        for _ in range(10000):
            if l_length > l_sl:
                l_offset = numpy.random.randint(l_length - l_sl)
            else:
                l_offset = 0
            offset = l_offset * l_scale

            silence = numpy.squeeze(silence_data.resample(sr, index=offset, length=sl))
            if (~silence).sum() >= min_not_silence_length:
                break
        else:
            raise Exception('cannot pick not silence data')

        wave = wave_data.wave[offset:offset + sl]

        # local
        l_start, l_end = l_offset - l_pad, l_offset + l_sl + l_pad
        if l_start < 0 or l_end > l_length:
            shape = list(local_data.array.shape)
            shape[0] = l_sl + l_pad * 2
            local = numpy.ones(shape=shape, dtype=local_data.array.dtype) * padding_value
            if l_start < 0:
                p_start = -l_start
                l_start = 0
            else:
                p_start = 0
            if l_end > l_length:
                p_end = l_sl + l_pad * 2 - (l_end - l_length)
                l_end = l_length
            else:
                p_end = l_sl + l_pad * 2
            local[p_start:p_end] = local_data.array[l_start:l_end]
        else:
            local = local_data.array[l_start:l_end]

        # source module
        if l_pad > 0:
            log_f0 = local[l_pad:-l_pad, f0_index]
        else:
            log_f0 = local[:, f0_index]

        if only_noise_source:
            log_f0 = numpy.zeros_like(log_f0)

        source = generate_source(
            log_f0=log_f0,
            local_rate=int(local_data.rate),
            sampling_rate=sr,
        )
        source2 = generate_source(
            log_f0=log_f0,
            local_rate=int(local_data.rate),
            sampling_rate=sr,
        )

        return dict(
            wave=wave,
            silence=silence,
            local=local,
            source=source,
            source2=source2,
        )

    def make_input(
            self,
            wave_data: Wave,
            silence_data: SamplingData,
            local_data: SamplingData,
    ):
        return self.extract_input(
            sampling_length=self.sampling_length,
            wave_data=wave_data,
            silence_data=silence_data,
            local_data=local_data,
            local_padding_length=self.local_padding_length,
            min_not_silence_length=self.min_not_silence_length,
            f0_index=self.f0_index,
            only_noise_source=self.only_noise_source,
        )


class WavesDataset(BaseWaveDataset):
    def __init__(
            self,
            inputs: List[Union[Input, LazyInput]],
            sampling_length: int,
            local_padding_length: int,
            min_not_silence_length: int,
            f0_index: int,
            only_noise_source: bool,
    ) -> None:
        super().__init__(
            sampling_length=sampling_length,
            local_padding_length=local_padding_length,
            min_not_silence_length=min_not_silence_length,
            f0_index=f0_index,
            only_noise_source=only_noise_source,
        )
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInput):
            input = input.generate()

        return default_convert(self.make_input(
            wave_data=input.wave,
            silence_data=input.silence,
            local_data=input.local,
        ))


class SpeakerWavesDataset(Dataset):
    def __init__(self, wave_dataset: Dataset, speaker_ids: List[int]):
        assert len(wave_dataset) == len(speaker_ids)
        self.wave_dataset = wave_dataset
        self.speaker_ids = speaker_ids

    def __len__(self):
        return len(self.wave_dataset)

    def __getitem__(self, i):
        d = self.wave_dataset[i]
        d['speaker_id'] = numpy.array(self.speaker_ids[i], dtype=numpy.long)
        return default_convert(d)


def create_dataset(config: DatasetConfig):
    wave_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.input_wave_glob))}
    fn_list = sorted(wave_paths.keys())
    assert len(fn_list) > 0

    silence_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.input_silence_glob))}
    assert set(fn_list) == set(silence_paths.keys())

    local_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.input_local_glob))}
    assert set(fn_list) == set(local_paths.keys())

    if config.speaker_dict_path is not None:
        fn_each_speaker: Dict[str, List[str]] = json.load(open(config.speaker_dict_path))
        assert config.speaker_size == len(fn_each_speaker)

        speaker_ids = {
            fn: speaker_id
            for speaker_id, (_, fns) in enumerate(fn_each_speaker.items())
            for fn in fns
        }
        assert set(fn_list).issubset(set(speaker_ids.keys()))
    else:
        speaker_ids = None

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    num_test = config.num_test
    num_train = config.num_train if config.num_train is not None else len(fn_list) - num_test

    trains = fn_list[num_test:][:num_train]
    tests = fn_list[:num_test]

    def make_dataset(fns, for_evaluate=False):
        inputs = [
            LazyInput(
                path_wave=wave_paths[fn],
                path_silence=silence_paths[fn],
                path_local=local_paths[fn],
            )
            for fn in fns
        ]

        if not for_evaluate:
            sampling_length = config.sampling_length
            local_padding_length = config.local_padding_length
        else:
            sampling_length = int(config.evaluate_time_second * config.sampling_rate)
            local_padding_length = int(config.evaluate_local_padding_time_second * config.sampling_rate)

        dataset = WavesDataset(
            inputs=inputs,
            sampling_length=sampling_length,
            local_padding_length=local_padding_length,
            min_not_silence_length=config.min_not_silence_length,
            f0_index=config.f0_index,
            only_noise_source=config.only_noise_source,
        )

        if speaker_ids is not None:
            dataset = SpeakerWavesDataset(
                wave_dataset=dataset,
                speaker_ids=[speaker_ids[fn] for fn in fns],
            )

        if for_evaluate:
            dataset = ConcatDataset([dataset] * config.evaluate_times)

        return dataset

    return dict(
        train=make_dataset(trains),
        test=make_dataset(tests),
        test_eval=make_dataset(tests, for_evaluate=True),
    )
