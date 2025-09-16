"""Utility helpers for persisting FinGraph artifacts.

This module centralizes artifact serialization so both the training
pipeline and the API share a consistent implementation.  The helpers are
purposefully lightweight – they only depend on the Python standard
library (plus ``numpy`` when it is available) so they can be reused in
scripts without pulling additional packages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Union

import numpy as np

LOGGER = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _ensure_path(path: PathLike) -> Path:
    """Return a resolved :class:`~pathlib.Path` for ``path``.

    The helper accepts ``Path`` objects or strings and always returns a
    concrete filesystem location.  Relative inputs are resolved against
    the current working directory rather than the module's location –
    this mirrors the behaviour of :func:`open` and keeps the functions
    intuitive for scripts that manage their own paths.
    """

    resolved = Path(path).expanduser()
    try:
        return resolved.resolve()
    except FileNotFoundError:
        # ``resolve`` raises when the path does not yet exist.  We still
        # want to return a sensible value so downstream code can create
        # the file or its parents.
        return resolved.absolute()


def _default_serializer(value: Any) -> Any:
    """JSON serializer for ``numpy`` types.

    ``numpy`` scalars/arrays are converted to their Python equivalents so
    that :func:`json.dump` can handle them.  Other unsupported values
    raise :class:`TypeError` to surface unexpected structures.
    """

    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Type {type(value)!r} is not JSON serialisable")


def load_artifact(path: PathLike) -> Any:
    """Load a persisted artifact.

    Parameters
    ----------
    path:
        Location of the artifact.  JSON files are deserialised using the
        standard library.  ``.joblib`` or ``.pkl`` files are loaded using
        :mod:`joblib` when it is available.  The helper deliberately does
        not hide :class:`FileNotFoundError`; callers can handle the
        absence of artifacts as appropriate for their context.
    """

    artifact_path = _ensure_path(path)
    suffix = artifact_path.suffix.lower()

    if suffix == ".json":
        with artifact_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    if suffix in {".joblib", ".pkl"}:
        try:
            import joblib  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "joblib is required to load non-JSON artifacts"
            ) from exc

        return joblib.load(artifact_path)

    raise ValueError(f"Unsupported artifact format: {artifact_path.suffix}")


def save_artifact(path: PathLike, payload: Any) -> None:
    """Persist ``payload`` to ``path``.

    JSON targets are written with UTF-8 encoding and pretty formatting so
    that version-controlled artifacts remain readable.  The function
    creates parent directories as needed.
    """

    artifact_path = _ensure_path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = artifact_path.suffix.lower()
    if suffix == ".json":
        with artifact_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=_default_serializer)
        return

    if suffix in {".joblib", ".pkl"}:
        try:
            import joblib  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "joblib is required to save non-JSON artifacts"
            ) from exc

        joblib.dump(payload, artifact_path)
        return

    raise ValueError(f"Unsupported artifact format: {artifact_path.suffix}")


__all__ = ["load_artifact", "save_artifact"]
