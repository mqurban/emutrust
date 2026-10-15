"""Run a slow model over many parameter points, with checkpointing."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np

from .space import ParameterSpace, as_space

Model = Callable[[dict], Any]
PathLike = Union[str, "os.PathLike[str]"]


@dataclass
class TrainingSet:
    """Inputs ``X`` (n, n_params) and model outputs ``Y`` (n, n_outputs)."""

    X: np.ndarray
    Y: np.ndarray
    names: tuple[str, ...]
    seconds_per_run: float

    def __len__(self) -> int:
        return len(self.X)


def run_model(
    model: Model,
    points: np.ndarray,
    space: ParameterSpace | Mapping[str, tuple[float, float]],
    checkpoint: PathLike | None = None,
    progress: bool = True,
) -> TrainingSet:
    """Evaluate ``model`` at every row of ``points``.

    ``model`` is any Python function that takes a ``{name: value}`` dict and
    returns an array (or list, or number) of outputs. Every call must return
    the same number of outputs.

    If ``checkpoint`` is a file path, results are saved after every run, and
    calling ``run_model`` again with the same points resumes where it stopped
    instead of starting over. This matters when each run takes seconds or
    minutes and the session can crash or time out.
    """
    space = as_space(space)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[1] != space.dim:
        raise ValueError(
            f"points have {points.shape[1]} columns but the space has {space.dim} parameters"
        )
    if not np.all(space.contains(points)):
        raise ValueError("Some points lie outside the parameter ranges.")

    path = Path(checkpoint) if checkpoint is not None else None
    outputs: list[np.ndarray] = []
    elapsed = 0.0
    if path is not None and path.exists():
        outputs, elapsed = _load_checkpoint(path, points, space)
        if progress and outputs:
            print(f"Resuming from checkpoint: {len(outputs)}/{len(points)} runs already done.")

    n_out = outputs[0].size if outputs else None
    for i in range(len(outputs), len(points)):
        t0 = time.perf_counter()
        y = np.asarray(model(space.to_dict(points[i])), dtype=float).ravel()
        elapsed += time.perf_counter() - t0

        if n_out is None:
            n_out = y.size
        if y.size != n_out:
            raise ValueError(
                f"Run {i} returned {y.size} outputs, but earlier runs returned {n_out}."
            )
        if not np.all(np.isfinite(y)):
            raise ValueError(
                f"Run {i} returned non-finite values (NaN or inf) at {space.to_dict(points[i])}."
            )
        outputs.append(y)

        if path is not None:
            _save_checkpoint(path, points, outputs, elapsed, space)
        if progress:
            print(f"  run {i + 1}/{len(points)} done ({elapsed / len(outputs):.2f} s per run)")

    return TrainingSet(
        X=points.copy(),
        Y=np.vstack(outputs),
        names=space.names,
        seconds_per_run=elapsed / max(len(outputs), 1),
    )


def _save_checkpoint(path: Path, points, outputs, elapsed, space) -> None:
    # Write to a temporary file and then rename it, so a crash in the middle
    # of saving can never leave a half-written, corrupted checkpoint behind.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(
            f,
            points=points,
            outputs=np.vstack(outputs),
            elapsed=elapsed,
            names=np.array(space.names),
        )
    os.replace(tmp, path)


def _load_checkpoint(path: Path, points, space):
    with np.load(path) as data:
        saved_points = data["points"]
        saved_outputs = data["outputs"]
        elapsed = float(data["elapsed"])
        saved_names = tuple(str(n) for n in data["names"])
    if (
        saved_names != space.names
        or saved_points.shape != points.shape
        or not np.allclose(saved_points, points)
    ):
        raise ValueError(
            f"The checkpoint at {path} belongs to a different set of points or parameters. "
            "Use a new checkpoint path, or delete the old file to start over."
        )
    return [row for row in saved_outputs], elapsed


def load_runs(path: PathLike) -> TrainingSet:
    """Load completed runs from a ``run_model`` checkpoint file.

    Useful for analysing saved runs without importing or re-running the
    slow model. Only finished runs are returned, so a partly complete
    checkpoint gives a shorter training set.
    """
    with np.load(path) as data:
        outputs = data["outputs"]
        n = len(outputs)
        return TrainingSet(
            X=data["points"][:n].copy(),
            Y=outputs.copy(),
            names=tuple(str(name) for name in data["names"]),
            seconds_per_run=float(data["elapsed"]) / max(n, 1),
        )


__all__ = ["TrainingSet", "load_runs", "run_model"]
