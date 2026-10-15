"""emutrust: neural network emulators for slow scientific models, with
calibrated uncertainty you can trust in Bayesian inference."""

from .emulator import Emulator
from .runner import TrainingSet, run_model
from .space import ParameterSpace, latin_hypercube

__version__ = "0.1.0.dev0"

__all__ = [
    "Emulator",
    "ParameterSpace",
    "TrainingSet",
    "latin_hypercube",
    "run_model",
    "__version__",
]
