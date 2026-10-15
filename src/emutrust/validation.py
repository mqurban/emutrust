"""Check an emulator against model runs it has never seen."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .emulator import Emulator
from .runner import TrainingSet

#: Fraction of a Gaussian within 1 and 2 standard deviations of its mean.
GAUSSIAN_1SIGMA = 0.6827
GAUSSIAN_2SIGMA = 0.9545


@dataclass
class ValidationReport:
    """How accurate an emulator is, where it is weak, and whether its
    uncertainty is honest."""

    names: tuple[str, ...]
    error_metric: str
    errors: np.ndarray  # (n_points, n_outputs)
    pulls: np.ndarray  # (n_points, n_outputs): error / predicted std
    edge_distance: np.ndarray  # (n_points,): 0 on a boundary, 0.5 at the centre
    X: np.ndarray
    residuals: np.ndarray  # (n_points, n_outputs): prediction - truth, signed
    truth: np.ndarray  # (n_points, n_outputs)
    per_parameter: dict = field(default_factory=dict)

    # --------------------------------------------------------------- accuracy
    @property
    def point_errors(self) -> np.ndarray:
        """Largest error over all outputs, for each test point."""
        return self.errors.max(axis=1)

    @property
    def mean_error(self) -> float:
        return float(self.errors.mean())

    @property
    def median_error(self) -> float:
        return float(np.median(self.errors))

    @property
    def max_error(self) -> float:
        return float(self.errors.max())

    def worst_points(self, n: int = 5) -> list[tuple[dict, float]]:
        """The ``n`` test points with the largest error, worst first."""
        order = np.argsort(self.point_errors)[::-1][:n]
        return [
            ({k: float(v) for k, v in zip(self.names, self.X[i])}, float(self.point_errors[i]))
            for i in order
        ]

    # ------------------------------------------------------------ calibration
    @property
    def coverage_1sigma(self) -> float:
        """Fraction of true values within 1 predicted std (ideal: 68.3%)."""
        return float(np.mean(np.abs(self.pulls) < 1))

    @property
    def coverage_2sigma(self) -> float:
        """Fraction of true values within 2 predicted std (ideal: 95.4%)."""
        return float(np.mean(np.abs(self.pulls) < 2))

    @property
    def suggested_std_scale(self) -> float:
        """Factor to multiply the predicted std by so that 68.3% of true
        values fall within 1 std.

        Above 1 means the emulator is overconfident (its error bars are too
        small); below 1 means it is underconfident. Set it with
        ``emulator.std_scale *= report.suggested_std_scale``, ideally using
        a calibration set separate from the final test set.
        """
        return float(np.quantile(np.abs(self.pulls), GAUSSIAN_1SIGMA))

    # -------------------------------------------------------- error covariance
    def error_covariance(self) -> tuple[str, np.ndarray]:
        """The emulator's measured error pattern, for use in a likelihood.

        Returns ``(kind, M)``. For ``kind == "relative"`` (positive outputs),
        ``M[i, j]`` is the average of ``r_i * r_j`` over the test points,
        where ``r = (prediction - truth) / truth``; the emulator error
        covariance at a prediction ``mu`` is then ``mu_i * mu_j * M[i, j]``.
        For ``kind == "absolute"``, ``M`` is the covariance in output units.

        The average of ``r_i * r_j`` (rather than a covariance about the
        mean) keeps any systematic offset of the emulator in the error
        budget. Off-diagonal terms capture errors that are correlated
        between outputs, which an independent-error model would miss.
        """
        if self.error_metric == "relative":
            r = self.residuals / self.truth
            return "relative", r.T @ r / len(r)
        r = self.residuals
        return "absolute", r.T @ r / len(r)

    # ----------------------------------------------------------------- output
    def summary(self) -> str:
        unit = "%" if self.error_metric == "relative" else " (in output std units)"
        scale = 100 if self.error_metric == "relative" else 1

        def fmt(v):
            return f"{v * scale:.3g}{unit}"

        near = self.edge_distance < 0.05
        lines = [
            f"Validation on {len(self.X)} unseen points",
            "",
            "Accuracy",
            f"  mean error:    {fmt(self.mean_error)}",
            f"  median error:  {fmt(self.median_error)}",
            f"  worst error:   {fmt(self.max_error)}",
        ]
        if near.any() and (~near).any():
            lines += [
                f"  median worst-output error, points within 5% of an edge: "
                f"{fmt(np.median(self.point_errors[near]))}",
                f"  median worst-output error, interior points:             "
                f"{fmt(np.median(self.point_errors[~near]))}",
            ]
        lines += [
            "",
            "Uncertainty calibration",
            f"  truth within 1 std: {100 * self.coverage_1sigma:.1f}%  (ideal 68.3%)",
            f"  truth within 2 std: {100 * self.coverage_2sigma:.1f}%  (ideal 95.4%)",
            f"  suggested std_scale: {self.suggested_std_scale:.2f}  "
            f"({_verdict(self.suggested_std_scale)})",
            "",
            "Worst points",
        ]
        for params, err in self.worst_points(3):
            values = ", ".join(f"{k}={v:.4g}" for k, v in params.items())
            lines.append(f"  {fmt(err)} at {values}")
        lines += ["", "Median worst-output error across each parameter's range (low -> high)"]
        for name, (_, med) in self.per_parameter.items():
            lines.append(f"  {name:>12}: " + "  ".join(fmt(m) for m in med))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def plot(self, path=None):
        """Plot the calibration and the error across each parameter.

        Needs matplotlib. Returns the figure, and saves it if ``path`` is
        given.
        """
        import matplotlib.pyplot as plt

        n = len(self.names)
        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 3.6))
        ax = axes[0]
        bins = np.linspace(-4, 4, 41)
        ax.hist(np.clip(self.pulls.ravel(), -4, 4), bins=bins, density=True, color="#7aa6c2")
        g = np.linspace(-4, 4, 200)
        ax.plot(g, np.exp(-0.5 * g**2) / np.sqrt(2 * np.pi), "k--", label="calibrated")
        ax.set_xlabel("(prediction - truth) / predicted std")
        ax.set_title("Uncertainty calibration")
        ax.legend(fontsize=8)

        scale = 100 if self.error_metric == "relative" else 1
        label = "worst-output error (%)" if self.error_metric == "relative" else "error"
        for ax, name in zip(axes[1:], self.names):
            i = self.names.index(name)
            ax.scatter(self.X[:, i], self.point_errors * scale, s=8, color="#1f4e79", alpha=0.6)
            ax.set_yscale("log")
            ax.set_xlabel(name)
            ax.set_ylabel(label)
            ax.set_title(f"Error across {name}")
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=130)
        return fig


def _verdict(scale: float) -> str:
    if scale > 1.2:
        return "overconfident: error bars too small"
    if scale < 0.8:
        return "underconfident: error bars too large"
    return "well calibrated"


def validate(emulator: Emulator, test: TrainingSet, n_bins: int = 5) -> ValidationReport:
    """Compare an emulator's predictions with true model outputs.

    ``test`` must contain runs the emulator was **not** trained on, for
    example made with ``run_model`` on a separate sample of points.

    Errors are relative (``|prediction - truth| / |truth|``) when every true
    output is positive, and otherwise divided by each output's spread across
    the test set.
    """
    X = np.asarray(test.X, dtype=float)
    Y = np.asarray(test.Y, dtype=float)
    if tuple(test.names) != emulator.names:
        raise ValueError(f"Test parameters {test.names} do not match emulator {emulator.names}")
    if len(X) < 2:
        raise ValueError("Need at least 2 test points")

    mean, std = emulator.predict(X)
    diff = np.abs(mean - Y)
    if np.all(Y > 0):
        metric, errors = "relative", diff / Y
    else:
        spread = Y.std(axis=0)
        metric, errors = "scaled", diff / np.where(spread > 0, spread, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        pulls = np.where(std > 0, (mean - Y) / std, np.where(diff > 0, np.inf, 0.0))

    low, high = emulator.bounds
    width = np.where(high > low, high - low, 1.0)
    u = (X - low) / width
    edge = np.clip(np.minimum(u, 1 - u), 0, 0.5).min(axis=1)

    point_err = errors.max(axis=1)
    per_param = {}
    for i, name in enumerate(emulator.names):
        edges = np.linspace(low[i], high[i], n_bins + 1)
        idx = np.clip(np.digitize(X[:, i], edges) - 1, 0, n_bins - 1)
        med = np.array(
            [np.median(point_err[idx == b]) if np.any(idx == b) else np.nan for b in range(n_bins)]
        )
        per_param[name] = (edges, med)

    return ValidationReport(
        names=emulator.names,
        error_metric=metric,
        errors=errors,
        pulls=pulls,
        edge_distance=edge,
        X=X,
        residuals=mean - Y,
        truth=Y,
        per_parameter=per_param,
    )


__all__ = ["ValidationReport", "validate"]
