import os
from pathlib import Path
import shutil

import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import jax
import jax.numpy as jnp

from save.checkpoint import TrainStateFlaxCheckpointer


class DummyModule(nn.Module):
    @nn.compact
    def __call__(self, *args, **kwargs) -> jax.Array:
        return nn.Dense(32)(jnp.zeros((1, 32)))


def make_dummy_params() -> dict:
    return DummyModule().init(jax.random.key(0))


def make_dummy_trainstate(seed: int) -> TrainState:
    key1, key2 = jax.random.split(jax.random.key(seed))
    return TrainState.create(
        apply_fn=lambda x: x,
        params={
            "key0": jax.random.uniform(key1, (2, 5)),
            "key1": jax.random.uniform(key2, (3, 4)),
        },
        tx=optax.adam(1.0),
    )


def test_save():
    path = Path("./foo/checkpoint").resolve()
    state = make_dummy_trainstate(0)

    ckptr = TrainStateFlaxCheckpointer(path)
    ckptr.save(state, 0)

    assert os.path.isdir(path.joinpath("0"))
    assert path.joinpath("0").joinpath("checkpoint.safetensors").exists()

    shutil.rmtree("./foo")


def test_save_restore():
    path = Path("./foo/checkpoint").resolve()
    state = make_dummy_trainstate(0)

    ckptr = TrainStateFlaxCheckpointer(path)
    ckptr.save(state, 0)

    _state = make_dummy_trainstate(-1)
    _state = ckptr.restore(_state, 0)

    assert state.params.keys() == _state.params.keys()
    for key, value in state.params.items():
        assert jnp.array_equal(value, _state.params[key])

    shutil.rmtree("./foo")


def test_restore_last():

    state_0 = make_dummy_trainstate(0)
    state_1 = make_dummy_trainstate(1)
    state_2 = make_dummy_trainstate(2)

    path = Path("./foo/checkpoint").resolve()

    ckptr = TrainStateFlaxCheckpointer(path)
    ckptr.save(state_0, 0)
    ckptr.save(state_1, 1)
    ckptr.save(state_2, 2)

    _state_2 = make_dummy_trainstate(2)
    step, _state_2 = ckptr.restore_last(_state_2)
    assert step == 2
    for key, value in state_2.params.items():
        assert jnp.array_equal(value, _state_2.params[key])

    shutil.rmtree("./foo")
