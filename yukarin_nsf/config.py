from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from yukarin_nsf.utility import dataclass_utility
from yukarin_nsf.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetConfig:
    sampling_rate: int
    sampling_length: int
    input_wave_glob: str
    input_silence_glob: str
    input_local_glob: str
    local_padding_length: int
    min_not_silence_length: int
    f0_index: int
    volume_index: Optional[int]
    harmonic_num: int
    only_noise_source: bool
    speaker_dict_path: Optional[str]
    speaker_size: Optional[int]
    seed: int
    num_train: Optional[int]
    num_test: int
    evaluate_times: int
    evaluate_time_second: float
    evaluate_local_padding_time_second: float


@dataclass
class NetworkConfig:
    speaker_size: Optional[int]
    speaker_embedding_size: Optional[int]
    local_size: int
    local_scale: int
    local_layer_num: int
    condition_size: int
    neural_filter_type: str
    neural_filter_layer_num: int
    neural_filter_stack_num: Optional[int]
    neural_filter_hidden_size: int
    discriminator_type: Optional[str]
    discriminator_layer_num: Optional[int]
    discriminator_hidden_size: Optional[int]


@dataclass
class ModelConfig:
    eliminate_silence: bool
    use_stft_weight: bool
    stft_config: List[Dict[str, int]]
    discriminator_input_type: Optional[str]
    adversarial_loss_scale: Optional[float]


@dataclass
class TrainConfig:
    batchsize: int
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: Optional[int]
    step_shift: Optional[Dict[str, Any]]
    num_processes: Optional[int] = None
    optimizer: Dict[str, Any] = field(default_factory=dict(name="Adam"))
    discriminator_optimizer: Optional[Dict[str, Any]] = None


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    assert d["dataset"]["speaker_size"] == d["network"]["speaker_size"]

    assert all(
        set(o.keys()) == {"fft_size", "hop_length", "window_length"}
        for o in d["model"]["stft_config"]
    )

    if "neural_filter_type" not in d["network"]:
        d["network"]["neural_filter_type"] = "gru"

    if "neural_filter_stack_num" not in d["network"]:
        d["network"]["neural_filter_stack_num"] = None

    assert (d["network"]["neural_filter_type"] == "wavenet") == (
        d["network"]["neural_filter_stack_num"] is not None
    )

    if "discriminator_type" not in d["network"]:
        d["network"]["discriminator_type"] = None
    if "discriminator_layer_num" not in d["network"]:
        d["network"]["discriminator_layer_num"] = None
    if "discriminator_hidden_size" not in d["network"]:
        d["network"]["discriminator_hidden_size"] = None

    if "discriminator_input_type" not in d["model"]:
        d["model"]["discriminator_input_type"] = None
    if "adversarial_loss_scale" not in d["model"]:
        d["model"]["adversarial_loss_scale"] = None
    assert (d["network"]["discriminator_type"] is None) == (
        d["model"]["discriminator_input_type"] is None
    )
    assert (d["network"]["discriminator_type"] is None) == (
        d["model"]["adversarial_loss_scale"] is None
    )

    if "discriminator_optimizer" not in d["train"]:
        d["train"]["discriminator_optimizer"] = None
    assert (d["network"]["discriminator_type"] is None) == (
        d["train"]["discriminator_optimizer"] is None
    )

    if "only_noise_source" not in d["dataset"]:
        d["dataset"]["only_noise_source"] = False

    if "step_shift" not in d["train"]:
        d["train"]["step_shift"] = None

    if "harmonic_num" not in d["dataset"]:
        d["dataset"]["harmonic_num"] = 0

    if "use_stft_weight" not in d["model"]:
        d["model"]["use_stft_weight"] = False

    if "volume_index" not in d["dataset"]:
        d["dataset"]["volume_index"] = None
