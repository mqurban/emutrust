import numpy as np
import pytest

from emutrust import TrainingSet, ValidationReport, validate


@pytest.fixture(scope="module")
def report(toy_emulator, toy_validation_runs):
    return validate(toy_emulator, toy_validation_runs)


def test_report_shapes_and_ranges(report, toy_validation_runs):
    n, n_out = toy_validation_runs.Y.shape
    assert report.errors.shape == report.pulls.shape == (n, n_out)
    assert report.error_metric == "relative"
    assert 0 < report.median_error <= report.mean_error <= report.max_error
    assert 0 <= report.coverage_1sigma <= report.coverage_2sigma <= 1
    assert report.suggested_std_scale > 0
    assert np.all((report.edge_distance >= 0) & (report.edge_distance <= 0.5))


def test_errors_match_a_direct_calculation(report, toy_emulator, toy_validation_runs):
    mean, _ = toy_emulator.predict(toy_validation_runs.X)
    expected = np.abs(mean - toy_validation_runs.Y) / toy_validation_runs.Y
    np.testing.assert_allclose(report.errors, expected)


def test_worst_points_are_sorted(report):
    worst = report.worst_points(4)
    errs = [e for _, e in worst]
    assert errs == sorted(errs, reverse=True)
    assert errs[0] == pytest.approx(report.point_errors.max())
    assert set(worst[0][0]) == set(report.names)


def test_summary_and_plot(report, tmp_path):
    text = report.summary()
    for phrase in ["Accuracy", "Uncertainty calibration", "suggested std_scale", "Worst points"]:
        assert phrase in text
    import matplotlib

    matplotlib.use("Agg")
    path = tmp_path / "report.png"
    report.plot(path)
    assert path.stat().st_size > 0


def test_error_covariance_is_symmetric_and_positive(report):
    kind, M = report.error_covariance()
    assert kind == "relative"
    n_out = report.errors.shape[1]
    assert M.shape == (n_out, n_out)
    np.testing.assert_allclose(M, M.T)
    assert np.linalg.eigvalsh(M).min() > -1e-12


def test_suggested_scale_detects_overconfidence():
    # Errors twice the size of the claimed std: the scale should be about 2.
    rng = np.random.default_rng(0)
    pulls = 2 * rng.standard_normal((4000, 3))
    rep = ValidationReport(
        names=("a",),
        error_metric="relative",
        errors=np.abs(pulls),
        pulls=pulls,
        edge_distance=np.full(4000, 0.25),
        X=np.zeros((4000, 1)),
        residuals=pulls,
        truth=np.ones_like(pulls),
    )
    assert rep.suggested_std_scale == pytest.approx(2.0, rel=0.05)
    assert "overconfident" in rep.summary()


def test_scaled_metric_for_outputs_that_can_be_negative():
    from emutrust import Emulator

    x = np.linspace(-1, 1, 60)[:, None]
    train = TrainingSet(X=x, Y=np.hstack([x, x**2]), names=("x",), seconds_per_run=0.0)
    emu = Emulator.fit(train, n_members=2, hidden=(16,), epochs=100, patience=100, seed=0)
    xt = np.linspace(-0.9, 0.9, 20)[:, None]
    test = TrainingSet(X=xt, Y=np.hstack([xt, xt**2]), names=("x",), seconds_per_run=0.0)
    rep = validate(emu, test)
    assert rep.error_metric == "scaled"
    assert rep.error_covariance()[0] == "absolute"


def test_rejects_mismatched_parameters(toy_emulator):
    test = TrainingSet(X=np.zeros((3, 1)), Y=np.ones((3, 50)), names=("x",), seconds_per_run=0.0)
    with pytest.raises(ValueError, match="do not match"):
        validate(toy_emulator, test)
