import os
import shutil
from pathlib import Path

from save.serializable import SerializableDict, SerializableObject, save_file
from save.serializable import JSONSerializable
from save.serializable import PickleSerializable


def test_static_serializable_dict():
    # Tests if serializable_dict is properly overwritten

    class Foo(SerializableObject):
        serializable_dict = SerializableDict({"a": None})

        def __init__(self):
            self.a = 0

    foo = Foo()
    assert "a" in foo.serializable_dict.keys()


def test_json_serialize_unserialize():
    foo = {"a": 0, "b": "1"}
    path = Path("./foo")

    JSONSerializable.serialize(foo, path)
    _foo = JSONSerializable.unserialize(path)

    assert foo["a"] == _foo["a"]
    assert foo["b"] == _foo["b"]

    os.remove(path)


class PickleFoo:
    def __init__(self, a, b):
        self.a, self.b = a, b


def test_pickle_serialize_unserialize():
    foo = PickleFoo(0, "1")
    path = Path("./foo")

    PickleSerializable.serialize(foo, path)
    _foo = PickleSerializable.unserialize(path)

    assert foo.a == _foo.a
    assert foo.b == _foo.b

    os.remove(path)


def test_recursive_serialize_unserialize():
    class Bar(SerializableObject):
        serializable_dict = SerializableDict(
            {"a": JSONSerializable, "b": JSONSerializable}
        )

        def __init__(self, a=0, b="1"):
            self.a, self.b = a, b

    class Foo(SerializableObject):
        serializable_dict = SerializableDict({"bar": Bar})

        def __init__(self, bar=None) -> None:
            self.bar = bar

    foo = Foo(Bar(1, "2"))
    path = Path("./foo")

    save_file(foo, path)
    _foo = Foo.unserialize(path)

    assert Path("./foo/").exists()
    assert Path("./foo/bar/").exists()
    assert Path("./foo/bar/a").exists()
    assert Path("./foo/bar/b").exists()

    assert foo.bar.a == _foo.bar.a
    assert foo.bar.b == _foo.bar.b

    shutil.rmtree(path)
