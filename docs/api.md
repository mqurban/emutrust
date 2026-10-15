# API reference

Everything below is importable from the top-level package, for example `from emutrust import Emulator`. For a walkthrough, see the [user guide](guide.md).

## Parameter spaces

### `ParameterSpace(bounds)`

Named parameters, each with a `(low, high)` range, e.g. `ParameterSpace({"H0": (60, 75), "ns": (0.92, 1.0)})`.

| Member | Description |
|---|---|
| `names` | Tuple of parameter names, in order |
| `dim` | Number of parameters |
| `low`, `high` | Arrays of lower and upper bounds |
| `sample(n, method="lhs", seed=None)` | `n` points, shape `(n, dim)`. `method` is `"lhs"` (Latin hypercube) or `"random"` |
| `to_dict(x)` | One point as `{name: value}` |
| `contains(x)` | Boolean array: which points lie inside the ranges |
| `to_unit(x)`, `from_unit(u)` | Map between physical values and the unit cube `[0, 1]^dim` |

### `latin_hypercube(n, dim, rng=None)`

`n` Latin hypercube points in the unit cube: each dimension is split into `n` equal slices and each slice is used exactly once.

## Running the slow model

### `run_model(model, points, space, checkpoint=None, progress=True)`

Evaluates `model` (a function taking a `{name: value}` dict and returning an array) at every row of `points`, and returns a `TrainingSet`.

- Every call must return the same number of outputs, all finite; otherwise a `ValueError` names the offending run.
- With `checkpoint` (a file path), results are saved atomically after every run, and calling again with the same points resumes where it stopped. A checkpoint made for different points or parameters is refused.

### `load_runs(path)`

Loads the finished runs from a `run_model` checkpoint as a `TrainingSet`, without needing the model.

### `TrainingSet`

| Field | Description |
|---|---|
| `X` | Inputs, shape `(n_runs, n_params)` |
| `Y` | Outputs, shape `(n_runs, n_outputs)` |
| `names` | Parameter names |
| `seconds_per_run` | Measured average cost of one model evaluation |

## Emulators

### `Emulator.fit(training, ...)`

Trains an ensemble of neural networks on a `TrainingSet` and returns an `Emulator`.

| Argument | Default | Description |
|---|---|---|
| `n_members` | `5` | Number of networks in the ensemble |
| `hidden` | `(128, 128, 128)` | Hidden layer widths |
| `epochs` | `2000` | Maximum training epochs per member |
| `batch_size` | `64` | Mini-batch size |
| `learning_rate` | `1e-3` | Initial Adam learning rate (halved on plateaus) |
| `val_fraction` | `0.1` | Fraction of runs each member holds out for early stopping |
| `patience` | `200` | Epochs without improvement before stopping |
| `log_outputs` | `None` | Learn `log10` of the outputs; `None` decides automatically |
| `seed` | `None` | Random seed for reproducible training |
| `verbose` | `False` | Print each member's best validation loss |

### `Emulator` members

| Member | Description |
|---|---|
| `predict(x, return_std=True)` | Mean and std of the ensemble at one point (dict or 1-D array) or many points (2-D array), in the model's original units |
| `predict_members(x)` | Every member's prediction, shape `(n_members, n_points, n_outputs)` |
| `emu(x)` | Same as `predict(x, return_std=False)` |
| `std_scale` | Multiplies every predicted std (default `1.0`); set it from `ValidationReport.suggested_std_scale` |
| `bounds` | `(low, high)` of the training inputs; predictions outside are extrapolations |
| `names`, `n_members` | Parameter names and ensemble size |
| `save(path)`, `Emulator.load(path)` | Save to and load from a single file |

## Validation

### `validate(emulator, test, n_bins=5)`

Compares predictions with true outputs from runs the emulator was not trained on, and returns a `ValidationReport`. Errors are relative when every output is positive, and otherwise scaled by each output's spread.

### `ValidationReport` members

| Member | Description |
|---|---|
| `mean_error`, `median_error`, `max_error` | Summary of the errors over all outputs and points |
| `point_errors` | Largest error over the outputs, for each test point |
| `worst_points(n=5)` | The `n` worst test points as `(parameters, error)` pairs |
| `coverage_1sigma`, `coverage_2sigma` | Fraction of true values within 1 and 2 predicted std (ideal 68.3% and 95.4%) |
| `suggested_std_scale` | Factor that would make the 1-std coverage 68.3% |
| `error_covariance()` | The measured error pattern, used by `GaussianLikelihood` |
| `summary()` / `print(report)` | Text report |
| `plot(path=None)` | Calibration histogram and error across each parameter (needs matplotlib) |

## Likelihood

### `GaussianLikelihood(emulator, data, noise_std, emulator_error="ensemble", error_variance_kept=0.999)`

Gaussian log-likelihood of `data` given the emulator's prediction, with covariance equal to the noise plus the emulator's error.

| Argument | Description |
|---|---|
| `data` | Observed values, one per emulator output |
| `noise_std` | Standard deviation of the noise: one number or one per data point |
| `emulator_error` | A `ValidationReport` (recommended), `"ensemble"`, or `None` (treat the emulator as exact) |
| `error_variance_kept` | For a `ValidationReport`: fraction of the measured error variance described by its main correlated patterns; the rest is treated as independent |

| Member | Description |
|---|---|
| `like(theta)` / `log_prob(theta)` | Log-posterior with a flat prior over the training range; accepts one point or a batch (use `vectorize=True` in `emcee`) |
| `log_likelihood(theta)` | Same value; `-inf` outside the training range |
| `in_bounds(theta)` | Which points lie inside the training range |
| `ndim` | Number of parameters |
| `n_error_modes` | Number of correlated error patterns in use |

## Toy model

`emutrust.toy` provides `toy_spectrum(params)`, a cheap function shaped like a power spectrum, with parameter ranges `TOY_SPACE` and evaluation points `TOY_K`. It is used in the tests and examples, so every component can be checked on a problem with a known answer.
