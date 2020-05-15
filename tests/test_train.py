import torch
from retry import retry

from tests.utility import SignWaveDataset, train_support
from yukarin_nsf.config import ModelConfig
from yukarin_nsf.model import Model, Networks, DiscriminatorModel, DiscriminatorInputType
from yukarin_nsf.network.discriminator import Discriminator, DiscriminatorType
from yukarin_nsf.network.predictor import Predictor, NeuralFilterType


def _create_model(
        local_size: int,
        local_scale: int,
        speaker_size=0,
        discriminator_type: DiscriminatorType=None,
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
        discriminator=Discriminator(
            input_size=1 if discriminator_type == DiscriminatorType.wavegan else 2,
            hidden_size=16,
            layer_num=10,
        ) if discriminator_type is not None else None,
    )

    if discriminator_type is None:
        discriminator_input_type = None
    elif discriminator_type == DiscriminatorType.wavegan:
        discriminator_input_type = DiscriminatorInputType.gan
    elif discriminator_type == DiscriminatorType.cgan:
        discriminator_input_type = DiscriminatorInputType.cgan
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
        discriminator_input_type=discriminator_input_type,
        adversarial_loss_scale=1,
    )
    model = Model(model_config=model_config, networks=networks, local_padding_length=0)
    if discriminator_type is not None:
        discriminator_model = DiscriminatorModel(model_config=model_config, networks=networks, local_padding_length=0)
    else:
        discriminator_model = None
    return model, discriminator_model


@retry(tries=10)
def test_train():
    model, _ = _create_model(local_size=1, local_scale=40)
    dataset = SignWaveDataset(
        sampling_length=16000,
        sampling_rate=16000,
        local_padding_length=0,
        local_scale=40,
    )

    def first_hook(o):
        assert o['main/loss'].data > 2

    def last_hook(o):
        assert o['main/loss'].data < 2

    iteration = 500
    train_support(
        batch_size=8,
        use_gpu=True,
        model=model,
        discriminator_model=None,
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
            f'-iteration={iteration}'
            '.pth'
        ),
    )


@retry(tries=10)
def test_train_discriminator():
    model, discriminator_model = _create_model(
        local_size=1,
        local_scale=40,
        discriminator_type=DiscriminatorType.wavegan,
    )
    dataset = SignWaveDataset(
        sampling_length=16000,
        sampling_rate=16000,
        local_padding_length=0,
        local_scale=40,
    )

    def first_hook(o):
        assert o['main/loss'].data > 3
        assert 'discriminator/loss' in o

    def last_hook(o):
        assert o['main/loss'].data < 3

    iteration = 500
    train_support(
        batch_size=8,
        use_gpu=True,
        model=model,
        discriminator_model=discriminator_model,
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
            f'-discriminator_type={DiscriminatorType.wavegan}'
        ),
    )


@retry(tries=10)
def test_train_conditional_discriminator():
    model, discriminator_model = _create_model(
        local_size=1,
        local_scale=40,
        discriminator_type=DiscriminatorType.cgan,
    )
    dataset = SignWaveDataset(
        sampling_length=16000,
        sampling_rate=16000,
        local_padding_length=0,
        local_scale=40,
    )

    def first_hook(o):
        assert o['main/loss'].data > 3
        assert 'discriminator/loss' in o

    def last_hook(o):
        assert o['main/loss'].data < 3

    iteration = 500
    train_support(
        batch_size=8,
        use_gpu=True,
        model=model,
        discriminator_model=discriminator_model,
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
            f'-discriminator_type={DiscriminatorType.cgan}'
        ),
    )
