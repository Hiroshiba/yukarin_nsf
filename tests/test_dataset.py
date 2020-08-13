import numpy
import pytest

from yukarin_nsf.dataset import generate_source


@pytest.fixture()
def log_f0():
    return numpy.array([0, numpy.log(250), 0, numpy.log(500)], dtype=numpy.float32)


@pytest.fixture()
def volume():
    return numpy.array([0.1, 0.2, 0.3, 0.4], dtype=numpy.float32)


@pytest.fixture()
def local_rate():
    return 8000 / 128


@pytest.fixture()
def harmonic_num():
    return 2


@pytest.fixture()
def sampling_rate():
    return 8000


def test_generate(log_f0: numpy.ndarray, local_rate: int, sampling_rate: int):
    source, signal = generate_source(
        log_f0=log_f0,
        volume=None,
        local_rate=local_rate,
        harmonic_num=0,
        sampling_rate=sampling_rate,
    )

    for i, x in enumerate(numpy.split(source, len(log_f0))):
        if log_f0[i] == 0:
            continue

        freq = numpy.fft.rfftfreq(len(x), d=1 / sampling_rate)
        expect = numpy.argmin(numpy.abs(numpy.exp(log_f0[i]) - freq))

        amp = numpy.abs(numpy.fft.rfft(x))
        assert numpy.argmax(amp) == expect

    for i, x in enumerate(numpy.split(signal, len(log_f0))):
        if log_f0[i] == 0:
            assert numpy.all(x == 0)

        else:
            freq = numpy.fft.rfftfreq(len(x), d=1 / sampling_rate)
            expect = numpy.argmin(numpy.abs(numpy.exp(log_f0[i]) - freq))

            amp = numpy.abs(numpy.fft.rfft(x))
            assert numpy.argmax(amp) == expect


def test_generate_volume(
    log_f0: numpy.ndarray, volume: numpy.ndarray, local_rate: int, sampling_rate: int
):
    source, signal = generate_source(
        log_f0=log_f0,
        volume=volume,
        local_rate=local_rate,
        harmonic_num=0,
        sampling_rate=sampling_rate,
    )

    max_values = []
    for x in numpy.split(source, len(log_f0)):
        max_values.append(x.max())

    assert numpy.all(numpy.argsort(volume) == numpy.argsort(max_values))


def test_generate_harmonic(
    log_f0: numpy.ndarray, local_rate: int, harmonic_num: int, sampling_rate: int
):
    source, signal = generate_source(
        log_f0=log_f0,
        volume=None,
        local_rate=local_rate,
        harmonic_num=harmonic_num,
        sampling_rate=sampling_rate,
    )

    for i_signal, x in enumerate(numpy.split(signal, len(log_f0))):
        if log_f0[i_signal] == 0:
            assert numpy.all(x == 0)

        else:
            freq = numpy.fft.rfftfreq(len(x), d=1 / sampling_rate)
            expect = set(
                numpy.argmin(numpy.abs(numpy.exp(log_f0[i_signal]) * (i_h + 1) - freq))
                for i_h in range(harmonic_num + 1)
            )

            amp = numpy.abs(numpy.fft.rfft(x))
            assert set(numpy.argsort(amp)[-(harmonic_num + 1) :]) == expect
