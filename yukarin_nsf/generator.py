from pathlib import Path
from typing import Union

import numpy
import torch
from acoustic_feature_extractor.data.wave import Wave

from yukarin_nsf.config import Config
from yukarin_nsf.network.predictor import Predictor, create_predictor


class Generator(object):
    def __init__(
            self,
            config: Config,
            predictor: Union[Predictor, Path],
            use_gpu: bool,
    ) -> None:
        self.config = config
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

        self.sampling_rate = config.dataset.sampling_rate
        self.f0_index = config.dataset.f0_index

    def generate(
            self,
            local: Union[numpy.ndarray, torch.Tensor],
            source: Union[numpy.ndarray, torch.Tensor],
            speaker_id: Union[numpy.ndarray, torch.Tensor] = None,
            local_padding_length: int = 0,
    ):
        if isinstance(local, numpy.ndarray):
            local = torch.from_numpy(local)
        local = local.to(self.device)

        if isinstance(source, numpy.ndarray):
            source = torch.from_numpy(source)
        source = source.to(self.device)

        if speaker_id is not None:
            if isinstance(speaker_id, numpy.ndarray):
                speaker_id = torch.from_numpy(speaker_id)
            speaker_id = speaker_id.to(self.device)

        # generate
        with torch.no_grad():
            output = self.predictor(
                source=source,
                local=local,
                local_padding_length=local_padding_length,
                speaker_id=speaker_id,
            )

        output = output.cpu().numpy()
        return [
            Wave(wave=o, sampling_rate=self.sampling_rate)
            for o in output
        ]
