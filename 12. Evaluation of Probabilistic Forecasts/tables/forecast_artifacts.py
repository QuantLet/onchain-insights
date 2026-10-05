"""Portable, losslessly compressed storage for forecast result dictionaries."""

from __future__ import annotations

import datetime as _datetime
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np


_FORMAT_VERSION = 1
_COMMON_FILENAME = "compact_test_data.npz"
_SHARED_CANDIDATES = (
    "true",
    "u_grid",
    "seq",
    "feature_names",
    "feature_list",
    "features",
)


def _numpy_pickle_compat():
    """Allow NumPy 2 pickles to be read by NumPy 1 where possible."""
    if not hasattr(np, "_core"):
        core = types.ModuleType("numpy._core")
        core.__dict__.update(np.core.__dict__)
        sys.modules.setdefault("numpy._core", core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


def load_legacy_pickle(path):
    _numpy_pickle_compat()
    with open(path, "rb") as stream:
        return pickle.load(stream)


def _encode(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {
                "type": "object_array",
                "shape": list(value.shape),
                "items": [_encode(item, arrays) for item in value.flat],
            }
        array_name = "array_{:06d}".format(len(arrays))
        arrays[array_name] = value
        return {"type": "ndarray", "name": array_name}

    if isinstance(value, np.datetime64):
        return {"type": "numpy_datetime64", "value": str(value), "dtype": str(value.dtype)}
    if isinstance(value, np.timedelta64):
        return {"type": "numpy_timedelta64", "value": str(value), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays)

    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None and value is pd.NA:
        return {"type": "pandas_na"}
    if pd is not None and value is pd.NaT:
        return {"type": "pandas_nat"}
    if pd is not None and isinstance(value, pd.Timestamp):
        return {
            "type": "pandas_timestamp",
            "value": int(value.value),
            "tz": str(value.tz) if value.tz is not None else None,
        }
    if pd is not None and isinstance(value, pd.Timedelta):
        return {"type": "pandas_timedelta", "value": int(value.value)}

    if pd is not None and isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe",
            "columns": _encode(list(value.columns), arrays),
            "index": _encode(value.index, arrays),
            "data": [_encode(value.iloc[:, i].to_numpy(), arrays)
                     for i in range(value.shape[1])],
            "dtypes": [str(dtype) for dtype in value.dtypes],
        }
    if pd is not None and isinstance(value, pd.Series):
        return {
            "type": "series",
            "name": _encode(value.name, arrays),
            "index": _encode(value.index, arrays),
            "data": _encode(value.to_numpy(), arrays),
            "dtype": str(value.dtype),
        }
    if pd is not None and isinstance(value, pd.Index):
        if isinstance(value, pd.MultiIndex):
            return {
                "type": "multi_index",
                "items": _encode(list(value), arrays),
                "names": _encode(list(value.names), arrays),
            }
        return {
            "type": "index",
            "items": _encode(value.to_numpy(), arrays),
            "name": _encode(value.name, arrays),
            "class": type(value).__name__,
        }

    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": [[_encode(k, arrays), _encode(v, arrays)]
                      for k, v in value.items()],
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode(item, arrays) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_encode(item, arrays) for item in value]}
    if value is None or isinstance(value, (str, bool, int)):
        return {"type": "scalar", "value": value}
    if isinstance(value, float):
        if np.isnan(value):
            return {"type": "float", "value": "nan"}
        if np.isposinf(value):
            return {"type": "float", "value": "inf"}
        if np.isneginf(value):
            return {"type": "float", "value": "-inf"}
        return {"type": "scalar", "value": value}
    if isinstance(value, (_datetime.datetime, _datetime.date)):
        return {
            "type": "datetime",
            "value": value.isoformat(),
            "class": type(value).__name__,
        }
    if isinstance(value, _datetime.timedelta):
        return {
            "type": "timedelta",
            "microseconds": ((value.days * 86400 + value.seconds) * 1000000
                             + value.microseconds),
        }

    raise TypeError("Unsupported value in forecast artifact: {}".format(type(value).__name__))


def _decode(node, archive):
    kind = node["type"]
    if kind == "ndarray":
        return archive[node["name"]]
    if kind == "object_array":
        values = [_decode(item, archive) for item in node["items"]]
        result = np.empty(node["shape"], dtype=object)
        result.flat[:] = values
        return result
    if kind == "scalar":
        return node["value"]
    if kind == "float":
        return {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}[node["value"]]
    if kind == "dict":
        return {_decode(k, archive): _decode(v, archive) for k, v in node["items"]}
    if kind == "list":
        return [_decode(item, archive) for item in node["items"]]
    if kind == "tuple":
        return tuple(_decode(item, archive) for item in node["items"])
    if kind == "datetime":
        cls = _datetime.datetime if node["class"] == "datetime" else _datetime.date
        return cls.fromisoformat(node["value"])
    if kind in ("pandas_na", "pandas_nat", "pandas_timestamp", "pandas_timedelta"):
        import pandas as pd
        if kind == "pandas_na":
            return pd.NA
        if kind == "pandas_nat":
            return pd.NaT
        if kind == "pandas_timestamp":
            return pd.Timestamp(node["value"], unit="ns", tz=node["tz"])
        return pd.Timedelta(node["value"], unit="ns")
    if kind == "timedelta":
        return _datetime.timedelta(microseconds=node["microseconds"])
    if kind == "numpy_datetime64":
        return np.datetime64(node["value"]).astype(node["dtype"])
    if kind == "numpy_timedelta64":
        return np.timedelta64(node["value"]).astype(node["dtype"])

    import pandas as pd

    if kind == "index":
        items = _decode(node["items"], archive)
        name = _decode(node["name"], archive)
        if node["class"] == "DatetimeIndex":
            return pd.DatetimeIndex(items, name=name)
        if node["class"] == "TimedeltaIndex":
            return pd.TimedeltaIndex(items, name=name)
        return pd.Index(items, name=name)
    if kind == "multi_index":
        return pd.MultiIndex.from_tuples(
            _decode(node["items"], archive),
            names=_decode(node["names"], archive),
        )
    if kind == "series":
        result = pd.Series(
            _decode(node["data"], archive),
            index=_decode(node["index"], archive),
            name=_decode(node["name"], archive),
        )
        try:
            return result.astype(node["dtype"])
        except (TypeError, ValueError):
            return result
    if kind == "dataframe":
        columns = _decode(node["columns"], archive)
        index = _decode(node["index"], archive)
        series = [pd.Series(_decode(col, archive), index=index) for col in node["data"]]
        result = pd.concat(series, axis=1) if series else pd.DataFrame(index=index)
        result.columns = columns
        for i, dtype in enumerate(node["dtypes"]):
            try:
                result.iloc[:, i] = result.iloc[:, i].astype(dtype)
            except (TypeError, ValueError):
                pass
        return result
    raise ValueError("Unknown forecast artifact node type: {}".format(kind))


def save_compact_archive(path, value):
    """Write a compressed NPZ archive without Python-pickled object arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    tree = _encode(value, arrays)
    arrays["__metadata__"] = np.asarray(json.dumps({
        "format_version": _FORMAT_VERSION,
        "tree": tree,
    }, separators=(",", ":")))

    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_compact_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"].item()))
        if metadata.get("format_version") != _FORMAT_VERSION:
            raise ValueError("Unsupported compact forecast artifact version in {}".format(path))
        return _decode(metadata["tree"], archive)


def find_shared_archive(path):
    for parent in (Path(path).parent,) + tuple(Path(path).parents):
        candidate = parent / _COMMON_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_forecast_artifact(path):
    """Load a legacy pickle or a compact per-model archive plus shared data."""
    path = preferred_forecast_path(path)
    compact_path = path

    if compact_path.suffix == ".npz":
        model_data = load_compact_archive(compact_path)
        shared_path = find_shared_archive(compact_path)
        if shared_path is not None and shared_path != compact_path:
            shared_data = load_compact_archive(shared_path)
            shared_data.update(model_data)
            model_data = shared_data
        return model_data
    return load_legacy_pickle(compact_path)


def preferred_forecast_path(path):
    """Prefer compact data, falling back to its legacy pickle when necessary."""
    path = Path(path)
    if path.name.endswith(".compact.npz"):
        legacy = path.with_name(path.name[:-len(".compact.npz")] + ".pkl")
        if path.is_file() and (
            not legacy.is_file() or path.stat().st_mtime >= legacy.stat().st_mtime
        ):
            return path
        if legacy.is_file():
            return legacy
        raise FileNotFoundError("Neither forecast file exists: {} or {}".format(path, legacy))
    if path.suffix == ".pkl":
        sibling = path.with_name(path.stem + ".compact.npz")
        if sibling.is_file() and (
            not path.is_file() or sibling.stat().st_mtime >= path.stat().st_mtime
        ):
            return sibling
        if not path.is_file():
            raise FileNotFoundError("Neither forecast file exists: {} or {}".format(path, sibling))
    return path


def values_equal(left, right):
    """Exact equality for finding fields shared by every model output."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
        except (TypeError, ValueError):
            try:
                return np.array_equal(np.asarray(left), np.asarray(right))
            except (TypeError, ValueError):
                return False
    if isinstance(left, dict) and isinstance(right, dict):
        return (left.keys() == right.keys() and
                all(values_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(values_equal(a, b) for a, b in zip(left, right))
    try:
        result = left == right
        return bool(result) if not isinstance(result, np.ndarray) else bool(result.all())
    except (TypeError, ValueError):
        return False


def shared_candidate_keys():
    return _SHARED_CANDIDATES
