"""Serialize JAX/NumPy pytrees without pickling live Python objects.

Layout under ``directory``::

    tree.json     # nested structure tags + scalar leaves
    arrays.npz    # named ndarray leaves (paths joined by ``/``)
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_ARRAY = "array"
_DICT = "dict"
_LIST = "list"
_TUPLE = "tuple"
_NAMEDTUPLE = "namedtuple"
_SCALAR = "scalar"
_NONE = "none"

_SCALAR_TYPES = (bool, int, float, str, np.bool_, np.integer, np.floating)


def _is_array(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "__array__")
        and not isinstance(value, (str, bytes, bytearray))
    )


def _is_namedtuple(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields")


def _qualname(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_qualname(qualname: str) -> type:
    module_name, _, class_name = qualname.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"Invalid type qualname {qualname!r}.")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in class_name.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise TypeError(f"Resolved {qualname!r} is not a type: {obj!r}.")
    return obj


def _encode_dict_key(key: Any) -> dict[str, Any]:
    if isinstance(key, bool):
        # bool is a subclass of int; keep distinct.
        return {"type": "bool", "value": bool(key)}
    if isinstance(key, int):
        return {"type": "int", "value": int(key)}
    if isinstance(key, str):
        return {"type": "str", "value": key}
    raise TypeError(f"Unsupported dict key type {type(key)!r}: {key!r}.")


def _decode_dict_key(node: dict[str, Any]) -> Any:
    key_type = node["type"]
    value = node["value"]
    if key_type == "bool":
        return bool(value)
    if key_type == "int":
        return int(value)
    if key_type == "str":
        return str(value)
    raise ValueError(f"Unknown dict key type {key_type!r}.")


def _encode(
    value: Any,
    *,
    arrays: dict[str, np.ndarray],
    path: str,
) -> dict[str, Any]:
    if value is None:
        return {"kind": _NONE}
    if _is_array(value):
        # npz stores keys as archive member names; avoid `/`.
        key = (path if path else "_root").replace("/", ".")
        if key in arrays:
            raise ValueError(f"Duplicate array path {key!r}.")
        arrays[key] = np.asarray(value)
        return {"kind": _ARRAY, "key": key}
    if isinstance(value, dict):
        items = []
        for key, child in value.items():
            key_token = key if isinstance(key, str) else repr(key)
            child_path = f"{path}/{key_token}" if path else key_token
            items.append(
                {
                    "key": _encode_dict_key(key),
                    "value": _encode(child, arrays=arrays, path=child_path),
                }
            )
        return {"kind": _DICT, "items": items}
    if _is_namedtuple(value):
        fields = {}
        for name in value._fields:
            child_path = f"{path}/{name}" if path else name
            fields[name] = _encode(getattr(value, name), arrays=arrays, path=child_path)
        return {
            "kind": _NAMEDTUPLE,
            "type": _qualname(type(value)),
            "fields": fields,
        }
    if isinstance(value, list):
        children = [
            _encode(child, arrays=arrays, path=f"{path}/{index}" if path else str(index))
            for index, child in enumerate(value)
        ]
        return {"kind": _LIST, "items": children}
    if isinstance(value, tuple):
        children = [
            _encode(child, arrays=arrays, path=f"{path}/{index}" if path else str(index))
            for index, child in enumerate(value)
        ]
        return {"kind": _TUPLE, "items": children}
    if isinstance(value, _SCALAR_TYPES):
        if isinstance(value, (np.bool_, np.integer, np.floating)):
            py_value: Any = value.item()
        else:
            py_value = value
        return {"kind": _SCALAR, "value": py_value, "type": type(py_value).__name__}
    raise TypeError(
        f"Unsupported pytree leaf type {type(value)!r} at path {path!r}. "
        "Convert to ndarray / dict / list / tuple / scalar before saving."
    )


def _decode(node: dict[str, Any], *, arrays: dict[str, np.ndarray]) -> Any:
    kind = node["kind"]
    if kind == _NONE:
        return None
    if kind == _ARRAY:
        key = node["key"]
        if key not in arrays:
            raise KeyError(f"Missing array leaf {key!r} in arrays.npz.")
        return arrays[key]
    if kind == _DICT:
        out: dict[Any, Any] = {}
        for item in node["items"]:
            out[_decode_dict_key(item["key"])] = _decode(item["value"], arrays=arrays)
        return out
    if kind == _NAMEDTUPLE:
        cls = _resolve_qualname(node["type"])
        fields = {
            name: _decode(child, arrays=arrays) for name, child in node["fields"].items()
        }
        return cls(**fields)
    if kind == _LIST:
        return [_decode(child, arrays=arrays) for child in node["items"]]
    if kind == _TUPLE:
        return tuple(_decode(child, arrays=arrays) for child in node["items"])
    if kind == _SCALAR:
        value = node["value"]
        type_name = node.get("type", type(value).__name__)
        if type_name == "bool":
            return bool(value)
        if type_name == "int":
            return int(value)
        if type_name == "float":
            return float(value)
        if type_name == "str":
            return str(value)
        return value
    raise ValueError(f"Unknown tree node kind {kind!r}.")


def save_pytree(directory: str | Path, tree: Any) -> None:
    """Write ``tree`` under ``directory`` as ``tree.json`` + ``arrays.npz``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    encoded = _encode(tree, arrays=arrays, path="")
    with (directory / "tree.json").open("w", encoding="utf-8") as handle:
        json.dump(encoded, handle, indent=2, sort_keys=False)
    arrays_path = directory / "arrays.npz"
    if arrays:
        np.savez_compressed(arrays_path, **arrays)
    elif arrays_path.exists():
        arrays_path.unlink()


def load_pytree(directory: str | Path) -> Any:
    """Load a pytree previously written by :func:`save_pytree`."""
    directory = Path(directory)
    tree_path = directory / "tree.json"
    arrays_path = directory / "arrays.npz"
    if not tree_path.is_file():
        raise FileNotFoundError(f"Missing {tree_path}")
    with tree_path.open("r", encoding="utf-8") as handle:
        encoded = json.load(handle)
    arrays: dict[str, np.ndarray] = {}
    if arrays_path.is_file():
        with np.load(arrays_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    return _decode(encoded, arrays=arrays)


def load_pytree_as_jax(directory: str | Path) -> Any:
    """Load pytree and move every ndarray leaf onto the default JAX device."""
    import jax
    import jax.numpy as jnp

    tree = load_pytree(directory)

    def _to_jax(leaf: Any) -> Any:
        if isinstance(leaf, np.ndarray):
            return jnp.asarray(leaf)
        return leaf

    return jax.tree.map(_to_jax, tree)
