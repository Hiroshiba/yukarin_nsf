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
        paths = model_dir.glob(prefix + '*.pth')
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + '{}.pth'.format(iteration))
        assert model_path.exists()
    return model_path


def generate(
        model_dir: Path,
        model_iteration: Optional[int],
        model_config: Optional[Path],
        output_dir: Path,
        num_test: int,
        time_second: float,
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

    config.dataset.sampling_length = int(config.dataset.sampling_rate * time_second)
    dataset = create_dataset(config.dataset)['test']

    if isinstance(dataset, SpeakerWavesDataset):
        local_paths = [input.path_local for input in dataset.wave_dataset.inputs][:num_test]
    elif isinstance(dataset, WavesDataset):
        local_paths = [input.path_local for input in dataset.inputs][:num_test]
    else:
        raise Exception()

    for data, local_path in tqdm(zip(chunked(dataset, batch_size), chunked(local_paths, batch_size)), desc='generate'):
        data = convert.concat_examples(data)
        output = generator.generate(
            local=data['local'],
            source=data['source'],
            speaker_id=data['speaker_id'] if 'speaker_id' in data else None,
            local_padding_length=config.dataset.local_padding_length,
        )

        for wave, p in zip(output, local_path):
            wave.save(output_dir / (p.stem + '.wav'))

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
    parser.add_argument('--num_test', type=int, default=10)
    parser.add_argument('--time_second', type=float, default=1)
    parser.add_argument('--use_gpu', action='store_true')
    generate(**vars(parser.parse_args()))
