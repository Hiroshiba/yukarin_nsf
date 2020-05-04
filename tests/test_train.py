import torch
from retry import retry

from tests.utility import SignWaveDataset, train_support
from yukarin_nsf.config import ModelConfig
from yukarin_nsf.model import Model, Networks
from yukarin_nsf.network.predictor import Predictor, NeuralFilterType


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
            neural_filter_type=NeuralFilterType.wavenet,
            neural_filter_layer_num=10,
            neural_filter_stack_num=1,
            neural_filter_hidden_size=16,
        ),
    )

    model_config = ModelConfig(
        eliminate_silence=True,
        stft_config=[
            dict(
                fft_size=512,
                hop_length=80,
                window_length=320,
            ),
            dict(
                fft_size=128,
                hop_length=40,
                window_length=80,
            ),
            dict(
                fft_size=2048,
                hop_length=640,
                window_length=1920,
            ),
        ],
    )
    model = Model(model_config=model_config, networks=networks, local_padding_length=0)
    return model


@retry(tries=10)
def test_train():
    model = _create_model(local_size=1, local_scale=40)
    dataset = SignWaveDataset(
        sampling_length=16000,
        sampling_rate=16000,
        local_padding_length=0,
        local_scale=40,
    )

    trained_loss = 2

    def first_hook(o):
        assert o['main/loss'].data > trained_loss

    def last_hook(o):
        assert o['main/loss'].data < trained_loss

    iteration = 500
    train_support(
        batch_size=8,
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
