import warnings
from copy import copy
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from pytorch_trainer.iterators import MultiprocessIterator
from pytorch_trainer.training import extensions, Trainer
from tensorboardX import SummaryWriter
from torch import optim, nn

from yukarin_nsf.config import Config
from yukarin_nsf.dataset import create_dataset
from yukarin_nsf.evaluator import GenerateEvaluator
from yukarin_nsf.generator import Generator
from yukarin_nsf.model import Model, DiscriminatorModel, create_network
from yukarin_nsf.updater import Updater
from yukarin_nsf.utility.tensorboard_extension import TensorboardReport


def create_optimizer(optimizer_config: Dict[str, Any], model: nn.Module):
    cp: Dict[str, Any] = copy(optimizer_config)
    n = cp.pop('name').lower()

    if n == 'adam':
        optimizer = optim.Adam(model.parameters(), **cp)
    elif n == 'sgd':
        optimizer = optim.SGD(model.parameters(), **cp)
    else:
        raise ValueError(n)

    return optimizer


def create_trainer(
        config_dict: Dict[str, Any],
        output: Path,
):
    # config
    config = Config.from_dict(config_dict)
    config.add_git_info()

    output.mkdir(parents=True)
    with (output / 'config.yaml').open(mode='w') as f:
        yaml.safe_dump(config.to_dict(), f)

    # model
    device = torch.device('cuda')

    networks = create_network(config.network)
    model = Model(
        model_config=config.model,
        networks=networks,
        local_padding_length=config.dataset.local_padding_length,
    )
    model.to(device)

    if config.model.discriminator_input_type is not None:
        discriminator_model = DiscriminatorModel(
            model_config=config.model,
            networks=networks,
            local_padding_length=config.dataset.local_padding_length,
        )
        discriminator_model.to(device)
    else:
        discriminator_model = None

    # dataset
    def _create_iterator(dataset, for_train: bool):
        return MultiprocessIterator(
            dataset,
            config.train.batchsize,
            repeat=for_train,
            shuffle=for_train,
            n_processes=config.train.num_processes,
            dataset_timeout=300,
        )

    datasets = create_dataset(config.dataset)
    train_iter = _create_iterator(datasets['train'], for_train=True)
    test_iter = _create_iterator(datasets['test'], for_train=False)
    test_eval_iter = _create_iterator(datasets['test_eval'], for_train=False)

    warnings.simplefilter('error', MultiprocessIterator.TimeoutWarning)

    # optimizer
    optimizer = create_optimizer(config.train.optimizer, model)
    if config.train.discriminator_optimizer is not None:
        discriminator_optimizer = create_optimizer(config.train.discriminator_optimizer, discriminator_model)
    else:
        discriminator_optimizer = None

    # updater
    updater = Updater(
        iterator=train_iter,
        optimizer=optimizer,
        discriminator_model=discriminator_model,
        model=model,
        discriminator_optimizer=discriminator_optimizer,
        device=device,
    )

    # trainer
    trigger_log = (config.train.log_iteration, 'iteration')
    trigger_snapshot = (config.train.snapshot_iteration, 'iteration')
    trigger_stop = (config.train.stop_iteration, 'iteration') if config.train.stop_iteration is not None else None

    trainer = Trainer(updater, stop_trigger=trigger_stop, out=output)

    ext = extensions.Evaluator(test_iter, model, device=device)
    trainer.extend(ext, name='test', trigger=trigger_log)
    if discriminator_model is not None:
        ext = extensions.Evaluator(test_iter, discriminator_model, device=device)
        trainer.extend(ext, name='test', trigger=trigger_log)

    generator = Generator(config=config, predictor=networks.predictor, use_gpu=True)
    generate_evaluator = GenerateEvaluator(
        generator=generator,
        time_length=config.dataset.evaluate_time_second,
        local_padding_time_length=config.dataset.evaluate_local_padding_time_second,
    )
    ext = extensions.Evaluator(test_eval_iter, generate_evaluator, device=device)
    trainer.extend(ext, name='eval', trigger=trigger_snapshot)

    ext = extensions.snapshot_object(networks.predictor, filename='predictor_{.updater.iteration}.pth')
    trainer.extend(ext, trigger=trigger_snapshot)
    ext = extensions.snapshot_object(trainer, filename='trainer_{.updater.iteration}.pth')
    trainer.extend(ext, trigger=trigger_snapshot)
    if networks.discriminator is not None:
        ext = extensions.snapshot_object(networks.discriminator, filename='discriminator_{.updater.iteration}.pth')
        trainer.extend(ext, trigger=trigger_snapshot)

    trainer.extend(extensions.FailOnNonNumber(), trigger=trigger_log)
    trainer.extend(extensions.LogReport(trigger=trigger_log))
    trainer.extend(extensions.PrintReport(['iteration', 'main/loss', 'test/main/loss']), trigger=trigger_log)

    ext = TensorboardReport(writer=SummaryWriter(Path(output)))
    trainer.extend(ext, trigger=trigger_log)

    (output / 'struct.txt').write_text(repr(model))
    if discriminator_model is not None:
        (output / 'discriminator_struct.txt').write_text(repr(discriminator_model))

    if trigger_stop is not None:
        trainer.extend(extensions.ProgressBar(trigger_stop))

    return trainer
