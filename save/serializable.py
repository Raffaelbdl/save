from abc import ABC, abstractstaticmethod
from collections import UserDict
import json
import os
from pathlib import Path
import pickle
from typing import Any

SERIALIZABLE_DICT = "serializable_dict"


class Serializable(ABC):
    @abstractstaticmethod
    def serialize(self, object: Any, path: Path): ...
    @abstractstaticmethod
    def unserialize(self, path: Path) -> Any: ...


class SerializableDict(UserDict[str, Serializable]):
    """Dictionary where keys are attribute names
    and values are Serializer classes.

    Example:
        class Foo():
            def __init__(self):
                self.a = pickable_object
                self.b = params

                self.serializable_dict = SerializableDict(
                    {
                        "a": PickleSerializable,
                        "b": ParamsSerializable,
                    }
                )
    """


class SerializableObject:
    serializable_dict = SerializableDict()

    @classmethod
    def unserialize(cls, path: Path):
        path = Path(path).resolve()

        kwargs = {}
        for attr, serial in cls.serializable_dict.items():
            _p = path.joinpath(attr)
            kwargs[attr] = serial.unserialize(_p)

        return cls(**kwargs)


def save_file(
    object: Any,
    path: Path,
    serializable: Serializable | SerializableObject | None = None,
) -> None:
    path = Path(path).resolve()

    if isinstance(object, SerializableObject):
        os.makedirs(path, exist_ok=True)

    else:
        if not hasattr(object, SERIALIZABLE_DICT):
            serializable.serialize(object, path)
            return
        raise ValueError("No Serializable has been provided for object.")

    serializable_dict: SerializableDict = getattr(object, SERIALIZABLE_DICT)

    if not isinstance(serializable_dict, SerializableDict):
        raise TypeError("serializable_dict is not a SerializableDict.")

    for attr, serial in serializable_dict.items():
        _path = path.joinpath(attr)
        save_file(getattr(object, attr), _path, serial)


##### Collection of Serializable #####


class DummySerializable(Serializable):
    """Does nothing"""

    @staticmethod
    def serialize(object: Any, path: Path):
        return

    @staticmethod
    def unserialize(path: Path) -> Any:
        return


class JSONSerializable(Serializable):
    """Serializable using JSON."""

    @staticmethod
    def serialize(object: Any, path: Path):
        with path.open("w") as file:
            json.dump(object, file)

    @staticmethod
    def unserialize(path: Path) -> Any:
        with path.open("r") as file:
            return json.load(file)


class PickleSerializable(Serializable):
    """Serializable using Pickle."""

    @staticmethod
    def serialize(object: Any, path: Path):
        with path.open("wb") as file:
            pickle.dump(object, file)

    @staticmethod
    def unserialize(path: Path) -> Any:
        with path.open("rb") as file:
            return pickle.load(file)
