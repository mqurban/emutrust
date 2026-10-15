import numpy as np
import pytest

from emutrust import ParameterSpace, latin_hypercube


@pytest.mark.parametrize("n, dim", [(1, 1), (10, 1), (50, 3), (300, 5)])
def test_lhs_uses_every_stratum_exactly_once(n, dim):
    u = latin_hypercube(n, dim, rng=0)
    assert u.shape == (n, dim)
    for d in range(dim):
        strata = np.floor(u[:, d] * n).astype(int)
        assert sorted(strata) == list(range(n))


def test_lhs_stays_inside_unit_cube():
    u = latin_hypercube(200, 4, rng=1)
    assert np.all((u >= 0) & (u < 1))


def test_lhs_is_reproducible_with_a_seed():
    np.testing.assert_array_equal(latin_hypercube(20, 3, rng=7), latin_hypercube(20, 3, rng=7))
    assert not np.array_equal(latin_hypercube(20, 3, rng=7), latin_hypercube(20, 3, rng=8))


def test_lhs_covers_ranges_more_evenly_than_random():
    # The emptiest slice under random sampling is usually far emptier than
    # under LHS, where every slice is always used once.
    n, bins = 100, 10
    lhs = latin_hypercube(n, 1, rng=3)[:, 0]
    rnd = np.random.default_rng(3).random(n)
    assert np.histogram(lhs, bins=bins, range=(0, 1))[0].min() == n // bins
    assert np.histogram(rnd, bins=bins, range=(0, 1))[0].min() < n // bins


@pytest.mark.parametrize("bad", [(0, 0), (-1, 1, 2), (1, 0), (0, np.inf)])
def test_space_rejects_bad_bounds(bad):
    with pytest.raises(ValueError):
        ParameterSpace({"x": bad})


def test_space_rejects_empty():
    with pytest.raises(ValueError):
        ParameterSpace({})


def test_space_sample_respects_ranges_and_shape():
    space = ParameterSpace({"H0": (60, 75), "ns": (0.92, 1.0)})
    X = space.sample(300, seed=42)
    assert X.shape == (300, 2)
    assert np.all(space.contains(X))
    assert space.names == ("H0", "ns")
    assert space.dim == 2


def test_unit_mapping_round_trips():
    space = ParameterSpace({"a": (-3, 5), "b": (10, 20)})
    X = space.sample(25, seed=0)
    np.testing.assert_allclose(space.from_unit(space.to_unit(X)), X)


def test_random_method_and_unknown_method():
    space = ParameterSpace({"a": (0, 1)})
    assert space.sample(5, method="random", seed=0).shape == (5, 1)
    with pytest.raises(ValueError):
        space.sample(5, method="grid")


def test_to_dict():
    space = ParameterSpace({"a": (0, 1), "b": (0, 1)})
    assert space.to_dict([0.25, 0.5]) == {"a": 0.25, "b": 0.5}
    with pytest.raises(ValueError):
        space.to_dict([0.1])
