import os
from pathlib import Path
import shutil

import flax.linen as nn
import jax
import jax.numpy as jnp

from save.checkpoint import flatten_params, unflatten_params
from save.checkpoint import FlaxCheckpointer


class DummyModule(nn.Module):
    @nn.compact
    def __call__(self, *args, **kwargs) -> jax.Array:
        return nn.Dense(32)(jnp.zeros((1, 32)))


def make_dummy_params() -> dict:
    return DummyModule().init(jax.random.key(0))


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

    ckptr = FlaxCheckpointer(path)
    assert os.path.isdir(path)

    shutil.rmtree("./foo")


def test_save():
    path = Path("./foo/checkpoint").resolve()
    params = make_dummy_params()

    ckptr = FlaxCheckpointer(path)
    ckptr.save(params, 0)

    assert os.path.isdir(path.joinpath("0"))
    assert path.joinpath("0").joinpath("checkpoint.safetensors").exists()

    shutil.rmtree("./foo")


def test_save_restore():
    key1, key2 = jax.random.split(jax.random.key(0))
    path = Path("./foo/checkpoint").resolve()
    params = {
        "key_0": jax.random.uniform(key1, (32, 6)),
        "key_1": jax.random.uniform(key2, (4, 29, 1)),
    }

    ckptr = FlaxCheckpointer(path)
    ckptr.save(params, 0)

    _params = ckptr.restore(0)

    assert params.keys() == _params.keys()
    for key, value in params.items():
        assert jnp.array_equal(value, _params[key])

    shutil.rmtree("./foo")


def test_restore_last():
    key = jax.random.key(0)
    key, key1, key2 = jax.random.split(key, 3)
    params_0 = {
        "key_0": jax.random.uniform(key1, (32, 6)),
        "key_1": jax.random.uniform(key2, (4, 29, 1)),
    }
    key, key1, key2 = jax.random.split(key, 3)
    params_1 = {
        "key_0": jax.random.uniform(key1, (32, 6)),
        "key_1": jax.random.uniform(key2, (4, 29, 1)),
    }
    key, key1, key2 = jax.random.split(key, 3)
    params_2 = {
        "key_0": jax.random.uniform(key1, (32, 6)),
        "key_1": jax.random.uniform(key2, (4, 29, 1)),
    }

    path = Path("./foo/checkpoint").resolve()

    ckptr = FlaxCheckpointer(path)
    ckptr.save(params_0, 0)
    ckptr.save(params_1, 1)
    ckptr.save(params_2, 2)

    _params_2 = ckptr.restore_last()
    for key, value in params_2.items():
        assert jnp.array_equal(value, _params_2[key])

    shutil.rmtree("./foo")


def test_path_to_checkpoint():
    path = Path("./foo/checkpoint").resolve()
    ckptr = FlaxCheckpointer(path)

    assert (
        ckptr._path_to_checkpoint(0)
        == Path("./foo/checkpoint/0/checkpoint.safetensors").resolve()
    )

    shutil.rmtree("./foo")
