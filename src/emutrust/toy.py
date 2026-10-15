"""Cheap toy models with known answers, for testing and examples.

A method should first be checked on a problem where the right answer is
known exactly; only then is it worth trusting on a real, slow model.
"""

from __future__ import annotations

import numpy as np

#: Scales at which the toy spectrum is evaluated (arbitrary units).
TOY_K = np.logspace(-2, 1, 50)

#: Parameter ranges for the toy spectrum.
TOY_SPACE = {"amplitude": (0.5, 2.0), "slope": (-2.0, -0.5), "cutoff": (1.0, 5.0)}


def toy_spectrum(params: dict) -> np.ndarray:
    """A smooth curve shaped loosely like a matter power spectrum.

    ``P(k) = amplitude * k**slope * exp(-k / cutoff)``, evaluated at
    ``TOY_K``. Like a real power spectrum, it spans several orders of
    magnitude and changes shape smoothly with its parameters, but it costs
    microseconds instead of seconds to compute.
    """
    return params["amplitude"] * TOY_K ** params["slope"] * np.exp(-TOY_K / params["cutoff"])


__all__ = ["TOY_K", "TOY_SPACE", "toy_spectrum"]
