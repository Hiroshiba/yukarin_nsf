import torch
from retry import retry

from tests.utility import SignWaveDataset, train_support
from yukarin_nsf.config import ModelConfig
from yukarin_nsf.model import Model, Networks
from yukarin_nsf.network.predictor import Predictor


def _create_model(
        local_size: int,
        local_scale: int,
        speaker_size=0,
):
    networks = Networks(
        predictor=Predictor(
            speaker_size=speaker_size,
            speaker_embedding_size=4,
            local_size=local_size,
            local_scale=local_scale,
            local_layer_num=1,
            condition_size=5,
            neural_filter_layer_num=1,
            neural_filter_hidden_size=128,
        ),
    )

    model_config = ModelConfig(
        eliminate_silence=True,
        stft_config=[
            dict(
                fft_size=128,
                hop_length=80,
                window_length=100,
            ),
        ],
    )
    model = Model(model_config=model_config, networks=networks, local_padding_length=0)
    return model


@retry(tries=10)
def test_training():
    model = _create_model(local_size=0, local_scale=40)
    dataset = SignWaveDataset(
        sampling_length=4000,
        sampling_rate=8000,
        local_padding_length=0,
        local_scale=40,
    )

    trained_loss = 1

    def first_hook(o):
        assert o['main/loss'].data > trained_loss

    def last_hook(o):
        assert o['main/loss'].data < trained_loss

    iteration = 300
    train_support(
        batch_size=12,
        use_gpu=True,
        model=model,
        dataset=dataset,
        iteration=iteration,
        first_hook=first_hook,
        last_hook=last_hook,
    )

    # save model
    torch.save(
        model.predictor.state_dict(),
        (
            '/tmp/'
            f'test_training'
            f'-speaker_size=0'
            f'-iteration={iteration}.pth'
        ),
    )
