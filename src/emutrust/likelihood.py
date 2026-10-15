"""Likelihoods that include the emulator's own uncertainty."""

from __future__ import annotations

from typing import Union

import numpy as np

from .emulator import Emulator
from .validation import ValidationReport

_LOG_2PI = np.log(2 * np.pi)

EmulatorError = Union[str, ValidationReport, None]


class GaussianLikelihood:
    """Gaussian likelihood of observed data, with an emulator as the model.

    For parameters ``theta`` the emulator predicts the model output
    ``mu(theta)``. The likelihood compares the data with ``mu`` using a
    covariance that adds the data's noise and the emulator's own error::

        C = C_noise + C_emulator

    so an imperfect emulator widens the result instead of silently
    shrinking the error bars. ``emulator_error`` chooses ``C_emulator``:

    ``ValidationReport`` (recommended)
        The error pattern measured on validation runs the emulator never
        saw, including correlations between outputs. Emulator errors are
        usually systematic (too high at one output means too high at the
        next), and with many data points that correlated part dominates;
        this option captures it.
    ``"ensemble"``
        The spread of the ensemble members at ``theta``, treating each
        output's error as independent. Needs no extra model runs, but can
        underestimate correlated errors when there are many outputs.
    ``None``
        Treat the emulator as exact. Only for comparison: this is the
        overconfident likelihood that ``emutrust`` exists to avoid.

    Parameters outside the range the emulator was trained on get a
    log-likelihood of ``-inf``, so a sampler never relies on extrapolation.

    Example
    -------
    >>> report = validate(emu, validation_runs)
    >>> like = GaussianLikelihood(emu, data, noise_std, emulator_error=report)
    >>> sampler = emcee.EnsembleSampler(32, like.ndim, like, vectorize=True)
    """

    def __init__(
        self,
        emulator: Emulator,
        data: np.ndarray,
        noise_std: float | np.ndarray,
        emulator_error: EmulatorError = "ensemble",
    ):
        self.emulator = emulator
        self.data = np.asarray(data, dtype=float).ravel()
        n_out = emulator.config["n_outputs"]
        if self.data.size != n_out:
            raise ValueError(f"data has {self.data.size} values but the emulator predicts {n_out}")
        noise = np.broadcast_to(np.asarray(noise_std, dtype=float), self.data.shape)
        if np.any(noise <= 0):
            raise ValueError("noise_std must be positive")
        self.noise_var = noise**2

        if isinstance(emulator_error, ValidationReport):
            if emulator_error.names != emulator.names:
                raise ValueError("The validation report belongs to a different emulator")
            self._kind, self._error_matrix = emulator_error.error_covariance()
            self.emulator_error = "validation"
        elif emulator_error in ("ensemble", None):
            self.emulator_error = emulator_error
        else:
            raise ValueError("emulator_error must be a ValidationReport, 'ensemble' or None")
        self.low, self.high = emulator.bounds

    @property
    def ndim(self) -> int:
        return len(self.emulator.names)

    def in_bounds(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        return np.all((theta >= self.low) & (theta <= self.high), axis=1)

    def log_likelihood(self, theta: np.ndarray) -> np.ndarray | float:
        """Log-likelihood at one parameter vector, or at each row of many."""
        theta = np.asarray(theta, dtype=float)
        single = theta.ndim == 1
        theta = np.atleast_2d(theta)
        out = np.full(len(theta), -np.inf)
        ok = self.in_bounds(theta) & np.all(np.isfinite(theta), axis=1)
        if ok.any():
            out[ok] = self._log_gaussian(theta[ok])
        return float(out[0]) if single else out

    def log_prob(self, theta: np.ndarray) -> np.ndarray | float:
        """Log-posterior with a flat prior over the emulator's training range.

        Accepts one parameter vector or a batch of them, so it can be used
        with ``emcee.EnsembleSampler(..., vectorize=True)`` for speed.
        """
        return self.log_likelihood(theta)

    __call__ = log_prob

    def _log_gaussian(self, theta: np.ndarray) -> np.ndarray:
        mean, std = self.emulator.predict(theta)
        resid = self.data - mean

        if self.emulator_error != "validation":
            var = self.noise_var + (std**2 if self.emulator_error == "ensemble" else 0.0)
            return -0.5 * np.sum(resid**2 / var + np.log(var) + _LOG_2PI, axis=1)

        n = resid.shape[1]
        if self._kind == "relative":
            cov = mean[:, :, None] * mean[:, None, :] * self._error_matrix
        else:
            cov = np.broadcast_to(self._error_matrix, (len(theta), n, n)).copy()
        cov[:, np.arange(n), np.arange(n)] += self.noise_var
        chol = np.linalg.cholesky(cov)
        z = np.linalg.solve(chol, resid[..., None])[..., 0]
        logdet = 2 * np.sum(np.log(np.diagonal(chol, axis1=1, axis2=2)), axis=1)
        return -0.5 * (np.sum(z**2, axis=1) + logdet + n * _LOG_2PI)


__all__ = ["GaussianLikelihood"]
