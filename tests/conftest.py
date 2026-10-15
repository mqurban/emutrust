import pytest

from emutrust import Emulator, ParameterSpace, run_model
from emutrust.toy import TOY_SPACE, toy_spectrum

FAST = dict(n_members=3, hidden=(64, 64), epochs=400, patience=60)


@pytest.fixture(scope="session")
def toy_space():
    return ParameterSpace(TOY_SPACE)


@pytest.fixture(scope="session")
def toy_training(toy_space):
    return run_model(toy_spectrum, toy_space.sample(300, seed=0), toy_space, progress=False)


@pytest.fixture(scope="session")
def toy_emulator(toy_training):
    return Emulator.fit(toy_training, seed=0, **FAST)


@pytest.fixture(scope="session")
def toy_validation_runs(toy_space):
    return run_model(toy_spectrum, toy_space.sample(150, seed=11), toy_space, progress=False)
