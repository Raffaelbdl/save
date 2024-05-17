import os
from pathlib import Path
import shutil

from save.checkpoint import flatten_params, unflatten_params
from save.checkpoint import Checkpointer


def test_flatten_params():
    foo = {"a": {"aa": 0, "ab": {"aba": 1}}, "b": 0}
    _foo = flatten_params(foo)
    assert _foo == {"a.aa": 0, "a.ab.aba": 1, "b": 0}


def test_unflatten_params():
    foo = {"a.aa": 0, "a.ab.aba": 1, "b": 0}
    _foo = unflatten_params(foo)
    assert _foo == {"a": {"aa": 0, "ab": {"aba": 1}}, "b": 0}


def test_init():
    path = Path("./foo/checkpoint").resolve()

    ckptr = Checkpointer(path)
    assert os.path.isdir(path)

    shutil.rmtree("./foo")


def test_path_to_checkpoint():
    path = Path("./foo/checkpoint").resolve()
    ckptr = Checkpointer(path)

    assert (
        ckptr._path_to_checkpoint(0)
        == Path("./foo/checkpoint/0/checkpoint.safetensors").resolve()
    )

    shutil.rmtree("./foo")
