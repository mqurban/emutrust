"""emutrust on CAMB: accuracy, speed, and whether the error bars are honest.

Uses the runs made by ``generate.py`` (run that first). The script:

1. trains a 5-member emulator on 300 CAMB matter power spectra,
2. validates it on 300 other CAMB runs,
3. compares the cost of one likelihood evaluation with one CAMB call,
4. runs a coverage test: for simulated observations with known true
   cosmological parameters, how often does the 68% interval from MCMC
   contain the truth, with and without the emulator's error included?

Run from the repository root (needs camb, emcee and matplotlib):

    python examples/camb/demo.py
"""

import sys
import time
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np

from emutrust import Emulator, GaussianLikelihood, load_runs, validate

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from generate import SPACE, camb_power_spectrum  # noqa: E402

IMAGES = HERE.parents[1] / "docs" / "images"
NOISE_LEVELS = [0.02, 0.002]  # fractional noise per data point
N_REALIZATIONS = 30
N_WALKERS, N_STEPS, BURN_IN = 32, 2500, 1000

# ---------------------------------------------------------------- 1. train
training = load_runs(HERE / "data" / "training.npz")
validation = load_runs(HERE / "data" / "validation.npz")
print(f"{len(training)} training runs, {len(validation)} validation runs")
t0 = time.perf_counter()
emu = Emulator.fit(training, n_members=5, seed=0)
print(f"Emulator trained in {time.perf_counter() - t0:.0f} s\n")

# ------------------------------------------------------------- 2. validate
report = validate(emu, validation)
print(report.summary(), "\n")
report.plot(IMAGES / "camb_validation.png")

# ---------------------------------------------------------------- 3. speed
like = GaussianLikelihood(emu, validation.Y[0], 0.01 * validation.Y[0], emulator_error=report)
batch = SPACE.sample(N_WALKERS, seed=5)
like(batch)  # warm-up
t0 = time.perf_counter()
for _ in range(50):
    like(batch)
per_eval = (time.perf_counter() - t0) / (50 * N_WALKERS)
print(f"One CAMB call:                  {training.seconds_per_run:.3f} s")
print(f"One emulator likelihood eval:   {1e3 * per_eval:.3f} ms (batches of {N_WALKERS})")
print(f"Speed-up:                       {training.seconds_per_run / per_eval:,.0f}x")
n_evals = N_WALKERS * N_STEPS
print(
    f"An MCMC run of {n_evals:,} evaluations: {n_evals * per_eval:.0f} s with the emulator, "
    f"about {n_evals * training.seconds_per_run / 3600:.0f} h with CAMB in the loop\n"
)

# ------------------------------------------------------------- 4. coverage
modes = {"emulator treated as exact": None, "+ validation-measured error": report}
rng = np.random.default_rng(1)
np.random.seed(0)  # emcee draws from NumPy's global generator
coverage = {}
for noise in NOISE_LEVELS:
    hits = {name: [] for name in modes}
    for _ in range(N_REALIZATIONS):
        theta = SPACE.from_unit(0.15 + 0.7 * rng.random(SPACE.dim))
        truth = camb_power_spectrum(SPACE.to_dict(theta))
        data = truth * (1 + noise * rng.standard_normal(truth.size))
        start = theta + 1e-3 * (SPACE.high - SPACE.low) * rng.standard_normal(
            (N_WALKERS, SPACE.dim)
        )
        for name, error in modes.items():
            like = GaussianLikelihood(emu, data, noise * truth, emulator_error=error)
            sampler = emcee.EnsembleSampler(N_WALKERS, SPACE.dim, like, vectorize=True)
            sampler.run_mcmc(start, N_STEPS, progress=False)
            chain = sampler.get_chain(discard=BURN_IN, flat=True)
            lo, hi = np.percentile(chain, [15.87, 84.13], axis=0)
            hits[name].append((theta >= lo) & (theta <= hi))
    print(f"Noise {100 * noise:g}% per point: how often the 68% interval contains the truth")
    for name, h in hits.items():
        h = np.array(h)
        coverage[(noise, name)] = h.mean()
        per_param = ", ".join(f"{n} {100 * c:.0f}%" for n, c in zip(SPACE.names, h.mean(axis=0)))
        print(f"  {name:<28} {100 * h.mean():3.0f}%   ({per_param})")
    print()

fig, ax = plt.subplots(figsize=(7.5, 3.8))
colors = {"emulator treated as exact": "#c0392b", "+ validation-measured error": "#1f4e79"}
width = 0.38
for j, name in enumerate(modes):
    xs = np.arange(len(NOISE_LEVELS)) + (j - 0.5) * width
    vals = [100 * coverage[(n, name)] for n in NOISE_LEVELS]
    ax.bar(xs, vals, width, color=colors[name], label=name)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
ax.axhline(68.3, color="k", ls="--", lw=1.2, label="honest error bars (68%)")
ax.set_xticks(np.arange(len(NOISE_LEVELS)))
ax.set_xticklabels([f"{100 * n:g}% noise per data point" for n in NOISE_LEVELS])
ax.set_ylim(0, 105)
ax.set_ylabel("68% interval contains truth (%)")
ax.set_title(f"CAMB P(k), 5 parameters, {N_REALIZATIONS} simulated observations per noise level")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(IMAGES / "camb_coverage.png", dpi=130)
print(f"Saved {IMAGES / 'camb_validation.png'} and {IMAGES / 'camb_coverage.png'}")
