"""Parameter spaces and how to sample training points from them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union

import numpy as np

SeedLike = Union[int, np.random.Generator, None]


def latin_hypercube(n: int, dim: int, rng: SeedLike = None) -> np.ndarray:
    """Draw ``n`` Latin hypercube points in the unit cube ``[0, 1]^dim``.

    Each dimension is split into ``n`` equal slices ("strata"), and every
    slice is used by exactly one point. This guarantees that the full range
    of every parameter is covered, however many dimensions there are, which
    plain random sampling does not.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if dim < 1:
        raise ValueError(f"dim must be at least 1, got {dim}")
    rng = np.random.default_rng(rng)

    # One random position inside each stratum: stratum i covers [i/n, (i+1)/n).
    offsets = rng.random((n, dim))
    points = (np.arange(n)[:, None] + offsets) / n

    # Shuffle the strata independently per dimension, so the points are not
    # all lined up along the diagonal of the cube.
    for d in range(dim):
        points[:, d] = points[rng.permutation(n), d]
    return points


class ParameterSpace:
    """Named parameters, each with a ``(low, high)`` range.

    Example
    -------
    >>> space = ParameterSpace({"H0": (60.0, 75.0), "ns": (0.92, 1.00)})
    >>> X = space.sample(300, seed=42)   # shape (300, 2)
    """

    def __init__(self, bounds: Mapping[str, tuple[float, float]]):
        if not bounds:
            raise ValueError("A parameter space needs at least one parameter.")
        names, lows, highs = [], [], []
        for name, limits in bounds.items():
            if len(limits) != 2:
                raise ValueError(f"Parameter {name!r} needs (low, high), got {limits!r}")
            low, high = (float(v) for v in limits)
            if not (np.isfinite(low) and np.isfinite(high)):
                raise ValueError(f"Parameter {name!r} has a non-finite bound: {limits!r}")
            if not low < high:
                raise ValueError(f"Parameter {name!r} needs low < high, got {limits!r}")
            names.append(str(name))
            lows.append(low)
            highs.append(high)
        self.names: tuple[str, ...] = tuple(names)
        self.low = np.array(lows)
        self.high = np.array(highs)

    @property
    def dim(self) -> int:
        """Number of parameters."""
        return len(self.names)

    def __repr__(self) -> str:
        ranges = ", ".join(
            f"{n}=({lo:g}, {hi:g})" for n, lo, hi in zip(self.names, self.low, self.high)
        )
        return f"ParameterSpace({ranges})"

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        """Map points from the unit cube to physical parameter values."""
        u = np.asarray(u, dtype=float)
        return self.low + u * (self.high - self.low)

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        """Map physical parameter values to the unit cube."""
        x = np.asarray(x, dtype=float)
        return (x - self.low) / (self.high - self.low)

    def contains(self, x: np.ndarray) -> np.ndarray:
        """Return True for each point that lies inside the parameter ranges."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.all((x >= self.low) & (x <= self.high), axis=1)

    def sample(self, n: int, method: str = "lhs", seed: SeedLike = None) -> np.ndarray:
        """Draw ``n`` points, returned as an array of shape ``(n, dim)``.

        ``method`` is ``"lhs"`` (Latin hypercube, the default) or ``"random"``
        (uniform random, mainly useful for comparison).
        """
        rng = np.random.default_rng(seed)
        if method == "lhs":
            u = latin_hypercube(n, self.dim, rng)
        elif method == "random":
            if n < 1:
                raise ValueError(f"n must be at least 1, got {n}")
            u = rng.random((n, self.dim))
        else:
            raise ValueError(f"Unknown sampling method {method!r}; use 'lhs' or 'random'.")
        return self.from_unit(u)

    def to_dict(self, x: np.ndarray) -> dict[str, float]:
        """Turn one point into a ``{name: value}`` dictionary."""
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.dim:
            raise ValueError(f"Expected {self.dim} values, got {x.size}")
        return {name: float(v) for name, v in zip(self.names, x)}


def as_space(space: ParameterSpace | Mapping[str, tuple[float, float]]) -> ParameterSpace:
    """Accept either a ParameterSpace or a plain ``{name: (low, high)}`` dict."""
    return space if isinstance(space, ParameterSpace) else ParameterSpace(space)


__all__ = ["ParameterSpace", "as_space", "latin_hypercube"]
