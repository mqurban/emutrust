# emutrust user guide

This guide walks through using `emutrust` on your own slow model, explains the choices you will make along the way, and states what the tool can and cannot promise. For the full list of functions and arguments, see the [API reference](api.md).

## When to use it

Use `emutrust` when you have a model that is:

- **slow** (a fraction of a second to minutes per evaluation),
- **deterministic** (the same inputs always give the same outputs),
- **smooth** in a handful of input parameters (roughly 1 to 10), and
- needed **many thousands of times**, typically inside MCMC parameter inference.

Cosmology codes such as CAMB and CLASS are the motivating example, but any Python function that maps a few numbers to an array of outputs will do.

## The workflow

```
ParameterSpace ──▶ run_model ──▶ Emulator.fit ──▶ validate ──▶ GaussianLikelihood ──▶ MCMC
 (ranges)          (slow runs)    (train)          (check)      (honest errors)
```

### 1. Define the parameter space

```python
from emutrust import ParameterSpace

space = ParameterSpace({"H0": (60, 75), "omch2": (0.10, 0.14), "ns": (0.92, 1.00)})
```

**Choose the ranges carefully.** The emulator is only trustworthy inside them, and the likelihood refuses parameter values outside the range covered by the training runs. Make them wide enough to contain every value your inference might explore, but no wider: each extra unit of range spreads the same training runs more thinly.

### 2. Run the slow model

```python
from emutrust import run_model


def my_model(params):  # params is a dict, e.g. {"H0": 67.3, ...}
    return slow_code(**params)  # must return the same number of outputs every time


training = run_model(my_model, space.sample(300, seed=0), space, checkpoint="train.npz")
validation = run_model(my_model, space.sample(300, seed=1), space, checkpoint="valid.npz")
```

- **Use separate seeds** for training and validation, so the validation runs are genuinely unseen.
- **Always pass a checkpoint path** for slow models. If the run is interrupted, calling `run_model` again with the same points resumes where it stopped. `load_runs("train.npz")` reloads finished runs later without the model.
- **How many runs?** A few hundred training runs is a reasonable start for 3 to 6 parameters. The validation report (step 4) tells you whether that was enough; adding more runs where it reports large errors is the most effective improvement.

### 3. Train the emulator

```python
from emutrust import Emulator

emu = Emulator.fit(training, n_members=5, seed=0)
emu.save("emulator.pt")
```

`fit` trains a **deep ensemble**: several neural networks that start from different random weights. Their average is the prediction, and their disagreement is an estimate of the emulator's own uncertainty. Inputs and outputs are standardised automatically, and outputs that are all positive and span more than two orders of magnitude (such as power spectra) are learned in `log10` space.

Defaults that usually work: 5 members, three hidden layers of 128 units, early stopping. Training 300 runs with 200 outputs takes about a minute on a laptop CPU.

### 4. Validate on unseen runs

```python
from emutrust import validate

report = validate(emu, validation)
print(report)
report.plot("validation.png")
```

Read the report in three parts:

| Section | What to look for |
|---|---|
| **Accuracy** | Is the typical error small compared with the noise in your data? If not, add training runs. |
| **Error across each parameter's range** | Large errors at one end of a range show where to add runs. |
| **Uncertainty calibration** | About 68% of true values should fall within 1 predicted std and 95% within 2. `suggested_std_scale` gives the correction if not. |

### 5. Infer parameters with an honest likelihood

```python
import emcee
from emutrust import GaussianLikelihood

like = GaussianLikelihood(emu, data, noise_std, emulator_error=report)
sampler = emcee.EnsembleSampler(32, like.ndim, like, vectorize=True)
sampler.run_mcmc(start, 5000)
```

The likelihood adds the emulator's error to the data's noise. Choose its source with `emulator_error`:

| Option | Use it when |
|---|---|
| `report` (a `ValidationReport`) | **Recommended.** Uses the error measured on the validation runs, including how errors at different outputs move together. |
| `"ensemble"` | You have no validation runs. Uses the ensemble's spread, treating each output's error as independent. With many outputs this can underestimate the error. |
| `None` | Only for comparison: treats the emulator as exact, which is what `emutrust` is designed to avoid. |

Pass `vectorize=True` to `emcee`: the likelihood evaluates all walkers in one batch, which is much faster than one at a time.

## Why emulator errors need special care

Random noise in data partly cancels out: with 200 data points, the noise in their average is about 14 times smaller than in any one point. Emulator errors do not behave like this. An emulator that is 0.1% too high at one scale is usually too high at the neighbouring scales as well, so the error is shared across many outputs and does not average away.

That is why `emulator_error=report` matters: it measures these shared error patterns directly from runs where the right answer is known. The ensemble spread alone cannot capture them well, because a handful of networks can only describe a handful of error patterns.

## How do you know the error bars are honest?

Run a **coverage test**, as the [examples](../examples) do: simulate many observations with true parameters drawn from the prior, run the inference on each, and count how often the 68% interval contains the truth. Honest error bars contain it about 68% of the time; much less means overconfident, much more means over-cautious.

Two details make the test meaningful:

- **Draw the true parameters from the whole prior** (the emulator's training range), not only from the middle. A credible interval is only guaranteed to contain the truth 68% of the time on average over truths drawn from the prior.
- **Check that the MCMC has converged.** Run chains for many autocorrelation times (`emcee` recommends at least 50) and start walkers spread around, not exactly on, the truth.

## Performance tips

- **Set `OPENBLAS_NUM_THREADS=1`** (or `OMP_NUM_THREADS=1`) when running MCMC on a CPU. PyTorch and NumPy otherwise both start worker threads and slow each other down.
- **Keep the default `error_variance_kept=0.999`.** It describes the measured error by its main patterns, which is much faster than using the full matrix and changes log-likelihood differences near the best fit by well under 0.1.
- **A GPU is not needed.** The batches in MCMC are small, and a CPU is fast enough.

## Limitations

- **No extrapolation.** Outside the training range the ensemble members spread apart but can all be wrong in the same direction, so their spread no longer measures the error. The likelihood therefore rejects parameters outside that range.
- **One error estimate everywhere.** The validation-measured error is averaged over the whole parameter space, so it is slightly too generous where the emulator is better than average and slightly too small where it is worse. In the coverage tests this made the error bars mildly cautious overall.
- **Smooth, low-dimensional models.** Ensembles of small networks work well for a handful of parameters and smooth outputs. Highly irregular models or tens of parameters would need more training runs and possibly different architectures.
- **Gaussian noise.** `GaussianLikelihood` assumes independent Gaussian noise on each data point.
