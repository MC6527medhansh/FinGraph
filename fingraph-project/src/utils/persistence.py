"""Persistence helpers supporting JSON and binary formats."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import joblib  # type: ignore
except Exception:  # pragma: no cover - fallback when joblib unavailable
    joblib = None  # type: ignore


def dump_artifact(obj: Any, path: Path) -> None:
    """Persist object to disk using an appropriate serializer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".json":
        with path.open("w", encoding="utf-8") as file:
            json.dump(obj, file, indent=2, sort_keys=True)
        return

    if joblib is not None:  # pragma: no branch - simple runtime check
        joblib.dump(obj, path)
    else:
        with path.open("wb") as file:
            pickle.dump(obj, file)


def load_artifact(path: Path) -> Any:
    """Load persisted object using the matching serializer."""
    path = Path(path)

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    if joblib is not None and path.suffix in {".joblib", ".pkl"}:
        try:  # pragma: no cover - runtime compatibility check
            return joblib.load(path)
        except Exception:
            pass

    with path.open("rb") as file:
        return pickle.load(file)
