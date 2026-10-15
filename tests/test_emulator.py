import numpy as np
import pytest

from emutrust import Emulator, ParameterSpace, TrainingSet, run_model
from emutrust.toy import TOY_K, TOY_SPACE, toy_spectrum

FAST = dict(n_members=3, hidden=(64, 64), epochs=400, patience=60)


@pytest.fixture(scope="module")
def toy_training():
    space = ParameterSpace(TOY_SPACE)
    return run_model(toy_spectrum, space.sample(300, seed=0), space, progress=False)


@pytest.fixture(scope="module")
def toy_emulator(toy_training):
    return Emulator.fit(toy_training, seed=0, **FAST)


def test_learns_the_toy_spectrum(toy_emulator):
    space = ParameterSpace(TOY_SPACE)
    test = run_model(toy_spectrum, space.sample(100, seed=123), space, progress=False)
    mean, _ = toy_emulator.predict(test.X)
    frac_err = np.abs(mean - test.Y) / test.Y
    assert np.median(frac_err) < 0.02


def test_uses_log_outputs_for_spectra(toy_emulator):
    assert toy_emulator.log_outputs is True


def test_uncertainty_grows_away_from_training_data():
    # Train only on the left half of the range; the ensemble members should
    # agree there and disagree on the right half, where they never saw data.
    x = np.linspace(0.0, 0.5, 60)[:, None]
    y = np.sin(6 * x) + x
    training = TrainingSet(X=x, Y=y, names=("x",), seconds_per_run=0.0)
    emu = Emulator.fit(training, n_members=5, hidden=(32, 32), epochs=600, patience=100, seed=1)
    _, std_inside = emu.predict([0.25])
    _, std_outside = emu.predict([1.5])
    assert std_outside[0] > 5 * std_inside[0]


def test_prediction_shapes_and_inputs(toy_emulator):
    point = {"amplitude": 1.0, "slope": -1.0, "cutoff": 3.0}
    mean, std = toy_emulator.predict(point)
    assert mean.shape == std.shape == (TOY_K.size,)
    assert np.all(std > 0)

    mean2, std2 = toy_emulator.predict(np.array([[1.0, -1.0, 3.0], [1.5, -0.8, 2.0]]))
    assert mean2.shape == std2.shape == (2, TOY_K.size)
    np.testing.assert_allclose(mean2[0], mean, rtol=1e-6)
    np.testing.assert_allclose(toy_emulator(point), mean, rtol=1e-6)

    with pytest.raises(ValueError, match="Missing parameters"):
        toy_emulator.predict({"amplitude": 1.0})
    with pytest.raises(ValueError, match="Expected 3 parameters"):
        toy_emulator.predict([1.0, 2.0])


def test_save_and_load_give_identical_predictions(toy_emulator, tmp_path):
    path = tmp_path / "emu.pt"
    toy_emulator.save(path)
    loaded = Emulator.load(path)
    x = ParameterSpace(TOY_SPACE).sample(5, seed=9)
    np.testing.assert_allclose(loaded.predict_members(x), toy_emulator.predict_members(x))
    assert loaded.names == toy_emulator.names
    assert loaded.log_outputs == toy_emulator.log_outputs


def test_same_seed_gives_same_emulator(toy_training):
    small = dict(n_members=2, hidden=(16,), epochs=50, patience=50)
    a = Emulator.fit(toy_training, seed=5, **small)
    b = Emulator.fit(toy_training, seed=5, **small)
    x = ParameterSpace(TOY_SPACE).sample(3, seed=0)
    np.testing.assert_allclose(a.predict_members(x), b.predict_members(x))


def test_linear_outputs_when_values_can_be_negative():
    x = np.linspace(-1, 1, 40)[:, None]
    training = TrainingSet(X=x, Y=np.hstack([x, x**2]), names=("x",), seconds_per_run=0.0)
    emu = Emulator.fit(training, n_members=2, hidden=(16,), epochs=50, patience=50, seed=0)
    assert emu.log_outputs is False
    with pytest.raises(ValueError, match="strictly positive"):
        Emulator.fit(training, log_outputs=True, n_members=1, epochs=5)


def test_constant_output_column_is_handled():
    x = np.linspace(0, 1, 30)[:, None]
    training = TrainingSet(
        X=x, Y=np.hstack([x, np.full_like(x, 2.0)]), names=("x",), seconds_per_run=0.0
    )
    emu = Emulator.fit(training, n_members=2, hidden=(16,), epochs=100, patience=100, seed=0)
    mean, _ = emu.predict([0.5])
    assert np.isfinite(mean).all()
    assert mean[1] == 2.0


def test_rejects_bad_settings(toy_training):
    with pytest.raises(ValueError, match="n_members"):
        Emulator.fit(toy_training, n_members=0)
    with pytest.raises(ValueError, match="val_fraction"):
        Emulator.fit(toy_training, val_fraction=0.7)
    tiny = TrainingSet(X=np.zeros((5, 1)), Y=np.ones((5, 1)), names=("x",), seconds_per_run=0.0)
    with pytest.raises(ValueError, match="at least 10"):
        Emulator.fit(tiny)
