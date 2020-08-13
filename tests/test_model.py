import numpy
import pytest
import torch

from yukarin_nsf.dataset import generate_source
from yukarin_nsf.model import stft_weight


@pytest.fixture()
def log_f0():
    return numpy.array([0, numpy.log(250), 0, numpy.log(500)], dtype=numpy.float32)


@pytest.fixture()
def local_rate():
    return 8000 / 128


@pytest.fixture()
def harmonic_num():
    return 2


@pytest.fixture()
def sampling_rate():
    return 8000


def test_stft_weight(log_f0: numpy.ndarray, local_rate: int, sampling_rate: int):
    _, signal = generate_source(
        log_f0=log_f0,
        volume=None,
        local_rate=local_rate,
        harmonic_num=0,
        sampling_rate=sampling_rate,
    )
    signal = signal[numpy.newaxis]

    length = int(sampling_rate / local_rate)

    weight = stft_weight(
        signal=torch.from_numpy(signal),
        fft_size=length,
        hop_length=length,
        window_length=length,
    )[0]

    for i, w in enumerate(weight.numpy().T):
        if log_f0[i] == 0:
            assert numpy.all(w == 0)

        else:
            freq = numpy.fft.rfftfreq(length, d=1 / sampling_rate)
            expect = numpy.argmin(numpy.abs(numpy.exp(log_f0[i]) - freq))

            assert numpy.argmax(w) == expect
            assert numpy.all(0 <= w)
            assert numpy.all(w <= 1)
