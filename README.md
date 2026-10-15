# emutrust

**Neural network emulators for slow scientific models, with calibrated uncertainty you can trust in Bayesian inference.**

> **Status:** early development (`0.1.0.dev0`). Training-set generation is working and tested; the emulator, uncertainty and likelihood components are in progress. See the [roadmap](#roadmap).

## The problem

Many scientific models are slow: a single evaluation of a Boltzmann code such as CAMB takes seconds, and more complex models can take minutes. Bayesian parameter inference with MCMC needs tens of thousands of evaluations, so running the model directly inside the sampler can take days.

An **emulator** is a fast stand-in: run the slow model a few hundred times, train a neural network on the results, and use the network inside the sampler instead. This can cut inference from days to seconds.

But an emulator is only approximately right. If the inference treats it as exact, its errors silently shrink the error bars and the result becomes **overconfident**. `emutrust` is built around closing that gap:

1. **Smart sampling** of training points across the parameter space, with checkpointing so slow runs are never lost.
2. **Training** of a neural network emulator that also estimates **its own uncertainty**.
3. **A validation report** showing accuracy on held-out runs, the worst cases, and where in parameter space the emulator is weak.
4. **An honest likelihood** that adds the emulator's uncertainty to the data's noise, ready for samplers such as `emcee`.

It works with **any** Python function: a cosmology code, a simulation wrapper, or any other expensive model.

## Installation

```bash
git clone https://github.com/mqurban/emutrust
cd emutrust
pip install -e ".[dev]"
```

## What works today

### 1. Define the parameter space and sample training points

```python
from emutrust import ParameterSpace

space = ParameterSpace(
    {
        "H0": (60.0, 75.0),
        "omch2": (0.10, 0.14),
        "ns": (0.92, 1.00),
    }
)

X = space.sample(300, seed=42)  # Latin hypercube, shape (300, 3)
```

Points are drawn with **Latin hypercube sampling**: each parameter's range is split into 300 equal slices and every slice is used exactly once. This covers the full range of every parameter evenly, which plain random sampling does not, and avoids the exponential cost of a grid (10 values per parameter in 5 dimensions is already 100,000 runs).

### 2. Run the slow model, with checkpointing

```python
from emutrust import run_model


def my_model(params):
    # params is a dict, e.g. {"H0": 67.3, "omch2": 0.12, "ns": 0.965}
    return slow_calculation(params)  # returns an array of outputs


training = run_model(my_model, X, space, checkpoint="runs.npz")
training.X, training.Y  # inputs (300, 3) and outputs (300, n_outputs)
training.seconds_per_run  # measured cost of one model evaluation
```

Progress is saved after every run. If the session crashes after 200 runs, calling `run_model` again with the same points resumes at run 201. The checkpoint is written atomically, so an interruption during a save cannot corrupt it, and it refuses to resume if the points or parameters have changed.

`run_model` also checks every output: a run that returns NaN or infinity, or a different number of outputs from earlier runs, stops with a clear error instead of silently polluting the training set.

### Toy model for testing

`emutrust.toy.toy_spectrum` is a cheap function shaped like a matter power spectrum (spanning several orders of magnitude and varying smoothly with three parameters). It is used to test every component on a problem where the right answer is known exactly, before trusting it on a real, slow model.

## Roadmap

- [x] Parameter spaces and Latin hypercube sampling
- [x] Checkpointed model runner with output validation
- [ ] Neural network emulator (PyTorch) with input/output standardisation and log-space outputs
- [ ] Emulator uncertainty from a deep ensemble
- [ ] Validation report: held-out accuracy, worst cases, uncertainty calibration
- [ ] Likelihood that includes emulator uncertainty, ready for `emcee`
- [ ] Demonstration on CAMB: inference with and without emulator uncertainty, compared against the true parameters
- [ ] Documentation and PyPI release

## Related work

- **[CosmoPower](https://github.com/alessiospuriomancini/cosmopower)** (Spurio Mancini et al. 2022): neural network emulators specialised for cosmological power spectra.
- **[Emukit](https://github.com/emukit/emukit)**: general emulation and uncertainty quantification based on Gaussian processes.

`emutrust` aims to be a general, lightweight tool for *any* slow model, focused on making emulator uncertainty an explicit part of the inference.

## Development

```bash
pytest          # run the tests
ruff check .    # lint
```

## License

MIT. See [LICENSE](LICENSE).
