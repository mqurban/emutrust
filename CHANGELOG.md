# Changelog

## 0.1.1 (2026-10-16)

- Archived on Zenodo so each release has a citable DOI.
- Added author ORCID to the citation metadata.
- Added a Colab notebook that walks through emutrust with plots.

## 0.1.0 (2026-10-15)

First release.

- `ParameterSpace` and `latin_hypercube` for sampling training points.
- `run_model` with atomic checkpointing and output validation; `load_runs` to reload saved runs.
- `Emulator`: deep-ensemble neural network emulator (PyTorch) with automatic input and output standardisation, automatic log-space outputs, early stopping, and save/load.
- `validate` and `ValidationReport`: accuracy, worst points, error across each parameter's range, uncertainty calibration, and the measured error covariance.
- `GaussianLikelihood`: adds emulator error (validation-measured or ensemble spread) to the data noise, rejects parameters outside the training range, and evaluates batches for vectorised `emcee` sampling.
- Coverage tests on a toy model and on CAMB matter power spectra.
