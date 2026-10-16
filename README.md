# emutrust

**Neural network emulators for slow scientific models, with calibrated uncertainty you can trust in Bayesian inference.**

[![tests](https://github.com/mqurban/emutrust/actions/workflows/tests.yml/badge.svg)](https://github.com/mqurban/emutrust/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/emutrust)](https://pypi.org/project/emutrust/)
[![Python](https://img.shields.io/pypi/pyversions/emutrust)](https://pypi.org/project/emutrust/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/mqurban/emutrust/blob/main/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mqurban/emutrust/blob/main/examples/emutrust_colab.ipynb)

> **Status:** alpha (`0.1.0`). All components are working, tested on Python 3.9 to 3.13, and demonstrated on CAMB. The API may still change before 1.0.

**Try it in your browser:** the [Colab notebook](https://colab.research.google.com/github/mqurban/emutrust/blob/main/examples/emutrust_colab.ipynb) walks through every step with plots, in about 5 minutes and with nothing to install.

**Documentation:** [user guide](https://github.com/mqurban/emutrust/blob/main/docs/guide.md) · [API reference](https://github.com/mqurban/emutrust/blob/main/docs/api.md) · [quickstart example](https://github.com/mqurban/emutrust/blob/main/examples/quickstart.py) · [changelog](https://github.com/mqurban/emutrust/blob/main/CHANGELOG.md)

## The problem

Many scientific models are slow: a single evaluation of a Boltzmann code such as CAMB takes seconds, and more complex models can take minutes. Bayesian parameter inference with MCMC needs tens of thousands of evaluations, so running the model directly inside the sampler can take days.

An **emulator** is a fast stand-in: run the slow model a few hundred times, train a neural network on the results, and use the network inside the sampler instead. This can cut inference from days to seconds.

But an emulator is only approximately right. If the inference treats it as exact, its errors silently shrink the error bars and the result becomes **overconfident**. `emutrust` is built around closing that gap:

1. **Smart sampling** of training points across the parameter space, with checkpointing so slow runs are never lost.
2. **Training** of a neural network emulator that also estimates **its own uncertainty**.
3. **A validation report** showing accuracy on held-out runs, the worst cases, and where in parameter space the emulator is weak.
4. **An honest likelihood** that adds the emulator's measured error to the data's noise, ready for samplers such as `emcee`.

It works with **any** Python function: a cosmology code, a simulation wrapper, or any other expensive model.

## Installation

```bash
pip install emutrust            # core: NumPy and PyTorch
pip install "emutrust[mcmc,plot]"  # also emcee and matplotlib, used in the examples
```

`emutrust` needs NumPy and PyTorch. On a machine without a GPU, installing the CPU-only build of PyTorch first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) avoids a large download.

## Usage

A complete, runnable version of the steps below is in [`examples/quickstart.py`](https://github.com/mqurban/emutrust/blob/main/examples/quickstart.py) (about 15 seconds on a laptop).

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

### 3. Train an emulator that knows its own uncertainty

```python
from emutrust import Emulator

emu = Emulator.fit(training, n_members=5, seed=0)

mean, std = emu.predict({"H0": 67.3, "omch2": 0.12, "ns": 0.965})
emu.save("emulator.pt")  # reload later with Emulator.load
```

`Emulator.fit` trains a **deep ensemble**: several neural networks on the same training set, each starting from different random weights and holding out a different random part of the data for early stopping. `predict` returns the ensemble average and the spread between members, both in the model's original units.

- **Where the training data pin the answer down, the members agree** and the spread is small.
- **Where data are thin or the function is hard to learn, they disagree**, and the spread grows. This spread is the emulator's estimate of its own error.

Inputs and outputs are standardised before training. Outputs that are all positive and span more than two orders of magnitude, such as power spectra, are learned in `log10` space automatically.

On the toy spectrum below, an ensemble of 5 networks trained on 300 runs gives, on 200 unseen test points ([`examples/toy_benchmark.py`](https://github.com/mqurban/emutrust/blob/main/examples/toy_benchmark.py)):

| Metric | Result | Ideal |
|---|---:|---:|
| Mean fractional error | 0.32% | |
| Median fractional error | 0.19% | |
| Worst fractional error | 12.8% | |
| Truth within 1 std of prediction | 67% | 68% |
| Truth within 2 std of prediction | 92% | 95% |

The last two rows test whether the uncertainty is *honest*: if it is well calibrated, the true value should fall within one predicted standard deviation 68% of the time, and within two 95% of the time. The worst cases are near the edges of the parameter space, where training points are sparsest.

**A known limitation.** Outside the parameter ranges used for training, the members spread apart, which correctly signals unfamiliar ground, but they can all extrapolate in the same wrong direction, so the spread no longer measures the true error ([`examples/ensemble_uncertainty.py`](https://github.com/mqurban/emutrust/blob/main/examples/ensemble_uncertainty.py)):

![Ensemble members agree inside the training range and spread apart outside it](https://raw.githubusercontent.com/mqurban/emutrust/main/docs/images/ensemble_uncertainty.png)

An emulator should therefore only be used inside the ranges it was trained on; the likelihood below enforces this.

### 4. Validate it on runs it has never seen

```python
from emutrust import validate

validation = run_model(my_model, space.sample(300, seed=1), space)
report = validate(emu, validation)
print(report)
report.plot("validation.png")
```

The report covers accuracy (mean, median and worst error), the worst parameter points, how the error varies across each parameter's range, and whether the uncertainty is honest. For an emulator trained on only 80 runs of the toy spectrum ([`examples/coverage_test.py`](https://github.com/mqurban/emutrust/blob/main/examples/coverage_test.py)), part of it reads:

```
Uncertainty calibration
  truth within 1 std: 65.1%  (ideal 68.3%)
  truth within 2 std: 87.6%  (ideal 95.4%)
  suggested std_scale: 1.08  (well calibrated)

Median worst-output error across each parameter's range (low -> high)
     amplitude: 3.65%  4.45%  4.06%  4.96%  4.68%
         slope: 4.16%  5.61%  4.68%  3.76%  3.34%
        cutoff: 14.7%  4.8%  4.68%  2.48%  3%
```

The last line points straight at the weak region: low `cutoff`, where the toy spectrum falls off most steeply. That is where more training runs would help most.

If the uncertainty is too small or too large, `report.suggested_std_scale` gives the correction, applied with `emu.std_scale *= report.suggested_std_scale`. Ideally use separate validation runs for this and for the final assessment.

### 5. Infer parameters with an honest likelihood

```python
import emcee
from emutrust import GaussianLikelihood

like = GaussianLikelihood(emu, data, noise_std, emulator_error=report)
sampler = emcee.EnsembleSampler(32, like.ndim, like, vectorize=True)
sampler.run_mcmc(start, 2000)
```

The likelihood compares the data with the emulator's prediction using a covariance `C = C_noise + C_emulator`:

- **`emulator_error=report`** (recommended): the emulator's error pattern measured on the validation runs, including **correlations between outputs**.
- **`emulator_error="ensemble"`**: the ensemble's spread at each point, with each output's error treated as independent.
- **`emulator_error=None`**: the emulator is treated as exact. This is the usual approach, kept for comparison.

Parameters outside the range the emulator was trained on are rejected, so the sampler never relies on extrapolation.

**Does it work? A coverage test.** [`examples/coverage_test.py`](https://github.com/mqurban/emutrust/blob/main/examples/coverage_test.py) simulates 30 observations with true parameters drawn from the prior (50 data points each, 1% noise), infers the parameters through a deliberately imperfect emulator (80 training runs), and counts how often the reported 68% interval contains the truth. Honest error bars contain it about 68% of the time:

![Coverage of the 68% interval for three likelihoods](https://raw.githubusercontent.com/mqurban/emutrust/main/docs/images/coverage.png)

| Likelihood | Truth inside the 68% interval |
|---|---:|
| Emulator treated as exact | 20% |
| + ensemble spread | 38% |
| + validation-measured error | 74% |

Treating the emulator as exact gives error bars far too small: the truth falls outside them most of the time. The ensemble spread helps, but not enough. The reason is that emulator errors are **systematic**: an emulator that is 1% too high at one scale is usually too high at the neighbouring scales too, so the error does not average away over 50 data points the way independent noise does. A 5-member ensemble can only describe a handful of such error patterns. The covariance measured on 300 validation runs captures them, and the result is close to honest (74% against the ideal 68%), slightly on the cautious side because a single error estimate is applied everywhere, including regions where the emulator is better than average.

With 30 observations of 3 parameters, each coverage figure is uncertain by about ±5 percentage points.

The true parameters are drawn from the whole prior, here the range the emulator was trained on. That is what makes 68% the right target: a Bayesian credible interval is only guaranteed to contain the truth 68% of the time *on average over truths drawn from the prior*. Drawing them only from the middle of the ranges inflates coverage whenever posteriors are wide enough to feel the edges.

For validation-measured error, the likelihood describes the measured error by its main correlated patterns (99.9% of the error variance by default, set with `error_variance_kept`) and treats the small remainder as independent per-output error. This is mathematically almost identical and much faster: on the CAMB example below, one evaluation takes about 0.3–0.5 ms instead of several milliseconds.

## Demonstration on CAMB

[`examples/camb/`](https://github.com/mqurban/emutrust/tree/main/examples/camb) applies `emutrust` to a real cosmology problem: the linear matter power spectrum P(k) from the Boltzmann code [CAMB](https://camb.info/), at 200 scales between k = 10⁻⁴ and 1 h/Mpc, over five ΛCDM parameters (H0, ombh2, omch2, ns, As). The setup matches the earlier [cosmo-emulator](https://github.com/mqurban/cosmo-emulator) project, which treated its emulator as exact.

- [`generate.py`](https://github.com/mqurban/emutrust/blob/main/examples/camb/generate.py) runs CAMB for 300 training and 300 validation points (CAMB 2.0.5). The runs are included in the repository, so CAMB itself is only needed to simulate new observations.
- [`demo.py`](https://github.com/mqurban/emutrust/blob/main/examples/camb/demo.py) trains a 5-member emulator, validates it, times it against CAMB and runs a coverage test.

**Accuracy and speed.**

| | Result |
|---|---:|
| Mean error on 300 unseen spectra | 0.05% |
| Median error | 0.03% |
| Worst error (near the edges of the parameter ranges) | 1.5% |
| Truth within 1 / 2 predicted std | 71% / 92% |
| One CAMB call | 0.58 s |
| One likelihood evaluation, including emulator error | 0.3–0.5 ms |
| One MCMC run (192,000 evaluations) | about 1 minute, against about 31 hours with CAMB in the loop |

The speed-up is roughly 1,500–2,000x; the exact figure varies with machine load.

![Validation of the CAMB emulator](https://raw.githubusercontent.com/mqurban/emutrust/main/docs/images/camb_validation.png)

**Are the error bars honest?** For 30 simulated observations at each of two noise levels, with true parameters drawn from the prior, MCMC (32 walkers, 6,000 steps, at least 47 autocorrelation times after burn-in) gives:

![Coverage for CAMB at two noise levels](https://raw.githubusercontent.com/mqurban/emutrust/main/docs/images/camb_coverage.png)

| Noise per data point | Emulator treated as exact | + validation-measured error |
|---|---:|---:|
| 2% | 75% | 75% |
| 0.2% | **55%** | **71%** |

(ideal: 68%; each figure is uncertain by about ±4 percentage points)

- **At 2% noise, the emulator's error does not matter.** The emulator is accurate to about 0.05%, forty times smaller than the noise, and both likelihoods agree. Treating this emulator as exact was safe for data of this quality.
- **At 0.2% noise, it does.** The emulator's systematic errors become comparable to the information in the data, and treating it as exact makes the error bars too small: the truth falls inside the 68% interval only 55% of the time. Including the validation-measured error restores honest error bars (71%).

As surveys reach sub-percent precision, an emulator that was good enough yesterday stops being good enough, and its error has to be part of the inference.

### Toy model for testing

`emutrust.toy.toy_spectrum` is a cheap function shaped like a matter power spectrum (spanning several orders of magnitude and varying smoothly with three parameters). It is used to test every component on a problem where the right answer is known exactly, before trusting it on a real, slow model.

## Roadmap

- [x] Parameter spaces and Latin hypercube sampling
- [x] Checkpointed model runner with output validation
- [x] Neural network emulator (PyTorch) with input/output standardisation and log-space outputs
- [x] Emulator uncertainty from a deep ensemble
- [x] Validation report: held-out accuracy, worst cases, uncertainty calibration
- [x] Likelihood that includes emulator error (ensemble or validation-measured), ready for `emcee`
- [x] Coverage test showing the effect on inference
- [x] Demonstration on CAMB: accuracy, speed, and coverage with and without emulator error
- [x] User guide, API reference and PyPI release
- [ ] Adaptive sampling: add training runs where the validation report shows the largest errors
- [ ] Error estimates that vary across the parameter space, instead of one average

## Related work

- **[CosmoPower](https://github.com/alessiospuriomancini/cosmopower)** (Spurio Mancini et al. 2022): neural network emulators specialised for cosmological power spectra.
- **[Emukit](https://github.com/emukit/emukit)**: general emulation and uncertainty quantification based on Gaussian processes.

`emutrust` aims to be a general, lightweight tool for *any* slow model, focused on making emulator uncertainty an explicit part of the inference.

## Development

```bash
git clone https://github.com/mqurban/emutrust
cd emutrust
pip install -e ".[dev]"
pytest          # run the tests
ruff check .    # lint
```

## License

MIT. See [LICENSE](https://github.com/mqurban/emutrust/blob/main/LICENSE).

## Citing

If `emutrust` is useful in your work, please cite it using the metadata in [`CITATION.cff`](https://github.com/mqurban/emutrust/blob/main/CITATION.cff).
