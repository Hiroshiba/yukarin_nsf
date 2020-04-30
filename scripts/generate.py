import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml
from more_itertools import chunked
from pytorch_trainer.dataset import convert
from tqdm import tqdm

from yukarin_nsf.config import Config
from yukarin_nsf.dataset import create_dataset, SpeakerWavesDataset, WavesDataset
from yukarin_nsf.generator import Generator
from utility.save_arguments import save_arguments


def _extract_number(f):
    s = re.findall(r'\d+', str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
        model_dir: Path,
        iteration: int = None,
        prefix: str = 'predictor_',
):
    if iteration is None:
        paths = model_dir.glob(prefix + '*.npz')
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + '{}.npz'.format(iteration))
        assert model_path.exists()
    return model_path


def generate(
        model_dir: Path,
        model_iteration: Optional[int],
        model_config: Optional[Path],
        output_dir: Path,
        use_gpu: bool,
):
    if model_config is None:
        model_config = model_dir / 'config.yaml'

    output_dir.mkdir(exist_ok=True)
    save_arguments(output_dir / 'arguments.yaml', generate, locals())

    config = Config.from_dict(yaml.safe_load(model_config.open()))

    model_path = _get_predictor_model_path(
        model_dir=model_dir,
        iteration=model_iteration,
    )
    generator = Generator(
        config=config,
        predictor=model_path,
        use_gpu=use_gpu,
    )

    batch_size = config.train.batchsize

    dataset = create_dataset(config.dataset)['test']

    if isinstance(dataset, SpeakerWavesDataset):
        local_paths = [input.path_local for input in dataset.wave_dataset.inputs]
    elif isinstance(dataset, WavesDataset):
        local_paths = [input.path_local for input in dataset.inputs]
    else:
        raise Exception()

    for obj in chunked(tqdm(zip(dataset, local_paths), desc='generate'), batch_size):
        data = convert.concat_examples([o[0] for o in obj])
        output = generator.generate(
            local=data['local'],
            source=data['source'],
            speaker_id=data['speaker_id'] if 'speaker_id' in data else None,
        )

        for wave, local_path in zip(output, [o[1] for o in obj]):
            wave.save(output_dir / (local_path.stem + '.wav'))

    # log_f0 = local[:, self.f0_index]
    # if isinstance(log_f0, torch.Tensor):
    #     log_f0 = log_f0.cpu().numpy()
    #
    # source = generate_source(
    #     log_f0=log_f0,
    #     local_rate=int(local_rate),
    #     sampling_rate=self.sampling_rate,
    # )
    # source = torch.from_numpy(source).to(self.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, type=Path)
    parser.add_argument('--model_iteration', type=int)
    parser.add_argument('--model_config', type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--use_gpu', action='store_true')
    generate(**vars(parser.parse_args()))
