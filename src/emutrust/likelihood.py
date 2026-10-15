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

    For validation-measured error, ``error_variance_kept`` sets how much of
    the measured error variance is described by its main correlated
    patterns (default 99%); the rest is kept as independent per-output
    error. This makes each evaluation much faster with a negligible change
    in the result. Use ``1.0`` to keep every pattern.

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
        error_variance_kept: float = 0.99,
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
            self._kind, matrix = emulator_error.error_covariance()
            self._set_error_modes(matrix, error_variance_kept)
            self.emulator_error = "validation"
        elif emulator_error in ("ensemble", None):
            self.emulator_error = emulator_error
        else:
            raise ValueError("emulator_error must be a ValidationReport, 'ensemble' or None")
        self.low, self.high = emulator.bounds

    def _set_error_modes(self, matrix: np.ndarray, kept: float) -> None:
        """Split the measured error covariance into its main patterns.

        The emulator's errors are dominated by a few patterns across the
        outputs. Keeping the largest eigenvectors that together explain a
        fraction ``kept`` of the error variance, and treating the small
        remainder as independent per-output error, gives almost exactly the
        same likelihood far faster: with ``K`` patterns and ``n`` outputs,
        each evaluation costs about ``n*K**2`` instead of ``n**3``. The
        remainder is kept on the diagonal, so every output's total error
        variance is preserved exactly.
        """
        if not 0 < kept <= 1:
            raise ValueError("error_variance_kept must be in (0, 1]")
        values, vectors = np.linalg.eigh(matrix)
        values, vectors = values[::-1], vectors[:, ::-1]
        values = np.clip(values, 0.0, None)
        total = values.sum()
        if total == 0:
            k = 0
        else:
            k = int(np.searchsorted(np.cumsum(values) / total, kept - 1e-12) + 1)
            k = min(k, int(np.sum(values > 1e-14 * values[0])))
        self._modes = vectors[:, :k]  # (n, K)
        self._mode_var = values[:k]  # (K,)
        kept_diag = np.sum(self._modes**2 * self._mode_var, axis=1)
        self._remainder_var = np.clip(np.diag(matrix) - kept_diag, 0.0, None)  # (n,)

    @property
    def n_error_modes(self) -> int:
        """Number of error patterns used (validation-measured error only)."""
        return self._modes.shape[1] if self.emulator_error == "validation" else 0

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

        # C = D + A diag(lam) A^T, with D diagonal (noise plus the small
        # remainder of the emulator error) and A the K main error patterns.
        # For relative errors the patterns scale with the prediction.
        n = resid.shape[1]
        scale = mean if self._kind == "relative" else np.ones_like(mean)
        d = self.noise_var + scale**2 * self._remainder_var  # (P, n)
        quad = np.sum(resid**2 / d, axis=1)
        logdet = np.sum(np.log(d), axis=1)
        if self._modes.shape[1] > 0:
            # Woodbury identity and matrix determinant lemma: only a K x K
            # matrix per point needs factorising.
            a = scale[:, :, None] * self._modes  # (P, n, K)
            a_d = a / d[:, :, None]
            small = np.einsum("pnk,pnj->pkj", a_d, a)
            k = small.shape[1]
            small[:, np.arange(k), np.arange(k)] += 1.0 / self._mode_var
            chol = np.linalg.cholesky(small)
            u = np.linalg.solve(chol, np.einsum("pnk,pn->pk", a_d, resid)[..., None])[..., 0]
            quad -= np.sum(u**2, axis=1)
            logdet += 2 * np.sum(np.log(np.diagonal(chol, axis1=1, axis2=2)), axis=1)
            logdet += np.sum(np.log(self._mode_var))
        return -0.5 * (quad + logdet + n * _LOG_2PI)


__all__ = ["GaussianLikelihood"]
