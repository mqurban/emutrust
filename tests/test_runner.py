import numpy as np
import pytest

from emutrust import ParameterSpace, load_runs, run_model
from emutrust.toy import TOY_K, TOY_SPACE, toy_spectrum


@pytest.fixture
def space():
    return ParameterSpace(TOY_SPACE)


def test_runs_every_point(space):
    X = space.sample(20, seed=0)
    ts = run_model(toy_spectrum, X, space, progress=False)
    assert ts.X.shape == (20, 3)
    assert ts.Y.shape == (20, TOY_K.size)
    assert ts.names == space.names
    np.testing.assert_allclose(ts.Y[3], toy_spectrum(space.to_dict(X[3])))


def test_accepts_plain_dict_space():
    X = ParameterSpace(TOY_SPACE).sample(5, seed=0)
    assert len(run_model(toy_spectrum, X, TOY_SPACE, progress=False)) == 5


def test_checkpoint_resumes_after_a_crash(space, tmp_path):
    X = space.sample(10, seed=1)
    ckpt = tmp_path / "runs.npz"
    calls = []

    def flaky(params):
        calls.append(params)
        if len(calls) == 6:
            raise RuntimeError("simulated crash")
        return toy_spectrum(params)

    with pytest.raises(RuntimeError):
        run_model(flaky, X, space, checkpoint=ckpt, progress=False)
    assert ckpt.exists()

    calls.clear()
    ts = run_model(toy_spectrum_counting(calls), X, space, checkpoint=ckpt, progress=False)
    assert len(calls) == 5  # only the 5 unfinished runs are repeated
    expected = np.vstack([toy_spectrum(space.to_dict(x)) for x in X])
    np.testing.assert_allclose(ts.Y, expected)


def toy_spectrum_counting(calls):
    def model(params):
        calls.append(params)
        return toy_spectrum(params)

    return model


def test_checkpoint_for_different_points_is_rejected(space, tmp_path):
    ckpt = tmp_path / "runs.npz"
    run_model(toy_spectrum, space.sample(5, seed=0), space, checkpoint=ckpt, progress=False)
    with pytest.raises(ValueError, match="different set of points"):
        run_model(toy_spectrum, space.sample(5, seed=1), space, checkpoint=ckpt, progress=False)


def test_rejects_inconsistent_output_sizes(space):
    sizes = iter([3, 3, 4])

    def model(params):
        return np.ones(next(sizes))

    with pytest.raises(ValueError, match="returned 4 outputs"):
        run_model(model, space.sample(3, seed=0), space, progress=False)


def test_rejects_non_finite_outputs(space):
    with pytest.raises(ValueError, match="non-finite"):
        run_model(lambda p: [1.0, np.nan], space.sample(2, seed=0), space, progress=False)


def test_rejects_points_outside_the_space(space):
    X = space.sample(3, seed=0)
    X[0, 0] = 100.0
    with pytest.raises(ValueError, match="outside"):
        run_model(toy_spectrum, X, space, progress=False)


def test_scalar_outputs_are_allowed():
    ts = run_model(lambda p: p["a"] ** 2, [[0.5], [0.2]], {"a": (0, 1)}, progress=False)
    np.testing.assert_allclose(ts.Y[:, 0], [0.25, 0.04])


def test_load_runs_reads_a_checkpoint(space, tmp_path):
    X = space.sample(8, seed=2)
    ckpt = tmp_path / "runs.npz"
    ts = run_model(toy_spectrum, X, space, checkpoint=ckpt, progress=False)
    loaded = load_runs(ckpt)
    np.testing.assert_array_equal(loaded.X, ts.X)
    np.testing.assert_array_equal(loaded.Y, ts.Y)
    assert loaded.names == space.names
    assert loaded.seconds_per_run >= 0


def test_load_runs_returns_only_finished_runs(space, tmp_path):
    X = space.sample(6, seed=3)
    ckpt = tmp_path / "runs.npz"
    calls = []

    def crash_after_two(params):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("stop")
        return toy_spectrum(params)

    with pytest.raises(RuntimeError):
        run_model(crash_after_two, X, space, checkpoint=ckpt, progress=False)
    loaded = load_runs(ckpt)
    assert len(loaded) == 2
    np.testing.assert_array_equal(loaded.X, X[:2])
