"""emutrust: neural network emulators for slow scientific models, with
calibrated uncertainty you can trust in Bayesian inference."""

from .emulator import Emulator
from .likelihood import GaussianLikelihood
from .runner import TrainingSet, load_runs, run_model
from .space import ParameterSpace, latin_hypercube
from .validation import ValidationReport, validate

__version__ = "0.1.1"

__all__ = [
    "Emulator",
    "GaussianLikelihood",
    "ParameterSpace",
    "TrainingSet",
    "ValidationReport",
    "latin_hypercube",
    "load_runs",
    "run_model",
    "validate",
    "__version__",
]
