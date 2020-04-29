import inspect
from pathlib import Path, PosixPath
from typing import Callable, Dict, Any

import yaml


def _path_represent(dumper, data):
    return dumper.represent_str(str(data))


yaml.SafeDumper.add_representer(PosixPath, _path_represent)


def save_arguments(path: Path, target_function: Callable, arguments: Dict[str, Any]):
    args = inspect.getfullargspec(target_function).args
    obj = {k: v for k, v in arguments.items() if k in args}

    with path.open(mode='w') as f:
        yaml.safe_dump(obj, f)
