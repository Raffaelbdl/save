import os
from pathlib import Path
from typing import Any

from flax.core import FrozenDict
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState
from safetensors.flax import save_file, load_file

FlaxParams = dict
ArrayDict = dict[str, Any]


def flatten_params(params: FlaxParams, key_prefix: str | None = None) -> ArrayDict:
    """
    Source:
        https://github.com/alvarobartt/safejax/blob/main/src/safejax/utils.py#L22

    Flatten a `Dict`, `FrozenDict`, or `VarCollection`, for more detailed information on
    the supported input types check `safejax.typing.ParamsDictLike`.

    Args:
        params: A `Dict` or `FrozenDict` with the params to flatten.
        key_prefix: A prefix to prepend to the keys of the flattened dictionary.

    Returns:
        A `Dict` containing the flattened params as level-1 key-value pairs.
    """
    flattened_params = {}
    for key, value in params.items():
        key = f"{key_prefix}.{key}" if key_prefix else key

        if isinstance(value, (dict, FrozenDict)):
            flattened_params.update(flatten_params(params=value, key_prefix=key))
        else:
            flattened_params[key] = value

    return flattened_params


def unflatten_params(params: ArrayDict) -> FlaxParams:
    """
    Source:
        https://github.com/alvarobartt/safejax/blob/main/src/safejax/utils.py#L69

    Unflatten a `Dict` where the keys should be expanded using the `.` character
    as a separator.

    Args:
        params: A `Dict` containing the params to unflatten by expanding the keys.

    Returns:
        An unflattened `Dict` where the keys are expanded using the `.` character.
    """
    unflattened_params = {}

    for key, value in params.items():
        unflattened_params_tmp = unflattened_params

        subkeys = key.split(".")
        for subkey in subkeys[:-1]:
            unflattened_params_tmp = unflattened_params_tmp.setdefault(subkey, {})

        unflattened_params_tmp[subkeys[-1]] = value

    return unflattened_params


class Checkpointer:
    def __init__(self, checkpoint_dir: os.PathLike) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _path_to_checkpoint(self, step: int) -> Path:
        return self.checkpoint_dir.joinpath(str(step)).joinpath(
            "checkpoint.safetensors"
        )


class FlaxCheckpointer(Checkpointer):

    def save(self, params: FlaxParams, step: int) -> None:
        flattened_params = flatten_params(params)

        path = self._path_to_checkpoint(step)
        os.makedirs(path.parent, exist_ok=True)

        save_file(flattened_params, path)

    def restore(self, step: int) -> FlaxParams:
        _path = self._path_to_checkpoint(step)

        if _path.exists():
            flattened_params = load_file(_path)
            return unflatten_params(flattened_params)

        raise FileNotFoundError(f"No checkpoint found at step {step}")

    def restore_last(self) -> tuple[int, FlaxParams]:
        checkpoints = [int(s) for s in os.listdir(self.checkpoint_dir)]
        checkpoints.sort()
        return checkpoints[-1], self.restore(checkpoints[-1])


class TrainStateFlaxCheckpointer(Checkpointer):

    def save(self, state: TrainState, step: int) -> None:
        flattened_params = flatten_params(state.params)

        path = self._path_to_checkpoint(step)
        os.makedirs(path.parent, exist_ok=True)

        save_file(flattened_params, path)

    def restore(self, state: TrainState, step: int) -> TrainState:
        _path = self._path_to_checkpoint(step)

        if not _path.exists():
            raise FileNotFoundError(f"No checkpoint found at step {step}")

        flattened_params = load_file(_path)
        return state.replace(params=unflatten_params(flattened_params))

    def restore_last(self, state: TrainState) -> tuple[int, TrainState]:
        checkpoints = [int(s) for s in os.listdir(self.checkpoint_dir)]
        checkpoints.sort()
        return checkpoints[-1], self.restore(state, checkpoints[-1])


class PyTreeNodeTrainStateFlaxCheckpointer(Checkpointer):

    def save(self, state: PyTreeNode, step: int) -> None:
        params = {key: value.params for key, value in state.__dict__.items()}
        flattened_params = flatten_params(params)

        path = self._path_to_checkpoint(step)
        os.makedirs(path.parent, exist_ok=True)

        save_file(flattened_params, path)

    def restore(self, state: PyTreeNode, step: int) -> TrainState:
        _path = self._path_to_checkpoint(step)

        if not _path.exists():
            raise FileNotFoundError(f"No checkpoint found at step {step}")

        flattened_params = load_file(_path)
        params = unflatten_params(flattened_params)

        for key, value in params.items():
            _train_state = getattr(state, key).replace(params=value)
            state = state.replace(**{key: _train_state})

        return state

    def restore_last(self, state: PyTreeNode) -> tuple[int, PyTreeNode]:
        checkpoints = [int(s) for s in os.listdir(self.checkpoint_dir)]
        checkpoints.sort()
        return checkpoints[-1], self.restore(state, checkpoints[-1])
