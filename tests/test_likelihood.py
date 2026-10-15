import numpy as np
import pytest

from emutrust import GaussianLikelihood, validate
from emutrust.toy import toy_spectrum

THETA = np.array([1.2, -1.1, 3.0])


@pytest.fixture(scope="module")
def data(toy_space):
    y = toy_spectrum(toy_space.to_dict(THETA))
    return y * (1 + 0.01 * np.random.default_rng(0).standard_normal(y.size)), 0.01 * y


def manual_gaussian(resid, cov):
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * (resid @ np.linalg.solve(cov, resid) + logdet + len(resid) * np.log(2 * np.pi))


def test_exact_mode_matches_the_gaussian_formula(toy_emulator, data):
    y, sig = data
    like = GaussianLikelihood(toy_emulator, y, sig, emulator_error=None)
    mean, _ = toy_emulator.predict(THETA)
    assert like(THETA) == pytest.approx(manual_gaussian(y - mean, np.diag(sig**2)))


def test_ensemble_mode_adds_the_emulator_variance(toy_emulator, data):
    y, sig = data
    like = GaussianLikelihood(toy_emulator, y, sig, emulator_error="ensemble")
    mean, std = toy_emulator.predict(THETA)
    assert like(THETA) == pytest.approx(manual_gaussian(y - mean, np.diag(sig**2 + std**2)))


def test_validation_mode_uses_the_measured_covariance(toy_emulator, toy_validation_runs, data):
    y, sig = data
    report = validate(toy_emulator, toy_validation_runs)
    like = GaussianLikelihood(toy_emulator, y, sig, emulator_error=report)
    mean, _ = toy_emulator.predict(THETA)
    _, M = report.error_covariance()
    cov = np.outer(mean, mean) * M + np.diag(sig**2)
    assert like(THETA) == pytest.approx(manual_gaussian(y - mean, cov))


def test_emulator_error_makes_a_poor_fit_less_unlikely(toy_emulator, data):
    # Away from the truth, the extra variance shrinks the chi-squared.
    y, sig = data
    far = THETA + np.array([0.05, 0.02, 0.1])
    exact = GaussianLikelihood(toy_emulator, y, sig, emulator_error=None)
    honest = GaussianLikelihood(toy_emulator, y, sig, emulator_error="ensemble")
    assert honest(far) - honest(THETA) > exact(far) - exact(THETA)


def test_outside_training_range_is_minus_infinity(toy_emulator, data):
    like = GaussianLikelihood(toy_emulator, *data)
    assert like(np.array([100.0, -1.0, 3.0])) == -np.inf
    assert like(np.array([np.nan, -1.0, 3.0])) == -np.inf


def test_batch_matches_single_calls(toy_emulator, toy_validation_runs, data):
    report = validate(toy_emulator, toy_validation_runs)
    like = GaussianLikelihood(toy_emulator, *data, emulator_error=report)
    batch = np.array([THETA, THETA + 0.01, [100.0, 0.0, 0.0]])
    out = like(batch)
    assert out.shape == (3,)
    for row, value in zip(batch, out):
        assert like(row) == pytest.approx(value)


def test_rejects_bad_inputs(toy_emulator, data):
    y, sig = data
    with pytest.raises(ValueError, match="values but the emulator predicts"):
        GaussianLikelihood(toy_emulator, y[:-1], sig[:-1])
    with pytest.raises(ValueError, match="positive"):
        GaussianLikelihood(toy_emulator, y, 0.0)
    with pytest.raises(ValueError, match="emulator_error"):
        GaussianLikelihood(toy_emulator, y, sig, emulator_error="full")
