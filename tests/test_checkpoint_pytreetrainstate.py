import os
from pathlib import Path
import shutil

import flax.linen as nn
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState
import optax
import jax
import jax.numpy as jnp

from save.checkpoint import PyTreeNodeTrainStateFlaxCheckpointer


class FooPyTree(PyTreeNode):
    foo: TrainState
    bar: TrainState


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


def make_dummy_pytreenode(seed=0) -> FooPyTree:
    return FooPyTree(
        foo=make_dummy_trainstate(seed), bar=make_dummy_trainstate(seed + 1)
    )


def test_save():
    path = Path("./foo/checkpoint").resolve()
    pytree = make_dummy_pytreenode()

    ckptr = PyTreeNodeTrainStateFlaxCheckpointer(path)
    ckptr.save(pytree, 0)

    assert os.path.isdir(path.joinpath("0"))
    assert path.joinpath("0").joinpath("checkpoint.safetensors").exists()

    shutil.rmtree("./foo")


def test_save_restore():
    path = Path("./foo/checkpoint").resolve()
    pytree = make_dummy_pytreenode()

    ckptr = PyTreeNodeTrainStateFlaxCheckpointer(path)
    ckptr.save(pytree, 0)

    _pytree = make_dummy_pytreenode()
    _pytree = ckptr.restore(_pytree, 0)

    assert pytree.foo.params.keys() == _pytree.foo.params.keys()
    assert pytree.bar.params.keys() == _pytree.bar.params.keys()

    for key, value in pytree.foo.params.items():
        assert jnp.array_equal(value, _pytree.foo.params[key])

    for key, value in pytree.bar.params.items():
        assert jnp.array_equal(value, _pytree.bar.params[key])

    shutil.rmtree("./foo")


def test_restore_last():
    pytree_0 = make_dummy_pytreenode(0)
    pytree_1 = make_dummy_pytreenode(1)
    pytree_2 = make_dummy_pytreenode(2)

    path = Path("./foo/checkpoint").resolve()

    ckptr = PyTreeNodeTrainStateFlaxCheckpointer(path)
    ckptr.save(pytree_0, 0)
    ckptr.save(pytree_1, 1)
    ckptr.save(pytree_2, 2)

    _pytree_2 = make_dummy_pytreenode(5)
    step, _pytree_2 = ckptr.restore_last(_pytree_2)
    assert step == 2

    for key, value in pytree_2.foo.params.items():
        assert jnp.array_equal(value, _pytree_2.foo.params[key])

    for key, value in pytree_2.bar.params.items():
        assert jnp.array_equal(value, _pytree_2.bar.params[key])

    shutil.rmtree("./foo")
