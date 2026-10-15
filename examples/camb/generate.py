"""Run CAMB to build training and validation sets for the CAMB demonstration.

The setup matches https://github.com/mqurban/cosmo-emulator: the linear
matter power spectrum P(k) at z = 0, at 200 log-spaced scales between
k = 1e-4 and 1 h/Mpc, over five LCDM parameters.

Run from the repository root (needs camb; about 0.6 s per CAMB call on a
modern CPU, so roughly 6 minutes in total):

    python examples/camb/generate.py

Runs are checkpointed, so an interrupted run resumes where it stopped.
"""

from pathlib import Path

import camb
import numpy as np

from emutrust import ParameterSpace, run_model

DATA = Path(__file__).parent / "data"

SPACE = ParameterSpace(
    {
        "H0": (60.0, 75.0),
        "ombh2": (0.020, 0.024),
        "omch2": (0.10, 0.14),
        "ns": (0.92, 1.00),
        "As": (1.8e-9, 2.4e-9),
    }
)
KH = np.logspace(-4, 0, 200)  # the k grid CAMB returns for these settings


def camb_power_spectrum(p: dict) -> np.ndarray:
    """Linear matter power spectrum P(k) at z = 0 from CAMB."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=p["H0"], ombh2=p["ombh2"], omch2=p["omch2"])
    pars.InitPower.set_params(ns=p["ns"], As=p["As"])
    pars.set_matter_power(redshifts=[0.0], kmax=1.1)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    _, _, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1.0, npoints=200)
    return pk[0]


if __name__ == "__main__":
    print(f"CAMB version {camb.__version__}")
    for name, n, seed in [("training", 300, 42), ("validation", 300, 1)]:
        print(f"\n{name}: {n} CAMB runs")
        runs = run_model(
            camb_power_spectrum,
            SPACE.sample(n, seed=seed),
            SPACE,
            checkpoint=DATA / f"{name}.npz",
            progress=False,
        )
        print(f"  done, {runs.seconds_per_run:.2f} s per CAMB call")
    (DATA / "camb_version.txt").write_text(camb.__version__ + "\n")
