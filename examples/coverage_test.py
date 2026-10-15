"""Do the error bars mean what they say? A coverage test.

We simulate many "observations" with known true parameters, infer the
parameters with MCMC through a deliberately imperfect emulator, and count
how often the reported 68% interval actually contains the truth. Honest
error bars contain it about 68% of the time.

Three likelihoods are compared:
  - emulator treated as exact
  - plus the ensemble's own spread
  - plus the emulator error measured on validation runs

Run from the repository root (about 5 minutes on a laptop CPU; needs emcee
and matplotlib):

    python examples/coverage_test.py
"""

import emcee
import matplotlib.pyplot as plt
import numpy as np

from emutrust import Emulator, GaussianLikelihood, ParameterSpace, run_model, validate
from emutrust.toy import TOY_SPACE, toy_spectrum

N_TRAIN = 80  # few training runs, so the emulator is noticeably imperfect
N_VALIDATION = 300
N_REALIZATIONS = 30
NOISE = 0.01  # 1% noise on each of the 50 data points

space = ParameterSpace(TOY_SPACE)
training = run_model(toy_spectrum, space.sample(N_TRAIN, seed=0), space, progress=False)
validation = run_model(toy_spectrum, space.sample(N_VALIDATION, seed=11), space, progress=False)
emu = Emulator.fit(training, n_members=5, seed=0)
report = validate(emu, validation)
print(report.summary(), "\n")

likelihoods = {
    "emulator treated as exact": None,
    "+ ensemble spread": "ensemble",
    "+ validation-measured error": report,
}
hits = {name: [] for name in likelihoods}
rng = np.random.default_rng(1)
np.random.seed(0)  # emcee draws from NumPy's global generator; seed it for reproducibility

for r in range(N_REALIZATIONS):
    # Draw the truth from the whole prior (the parameter ranges): a 68% credible
    # interval is only guaranteed to contain the truth 68% of the time on
    # average over truths drawn from the prior.
    theta = space.from_unit(rng.random(space.dim))
    truth = toy_spectrum(space.to_dict(theta))
    data = truth * (1 + NOISE * rng.standard_normal(truth.size))
    start = theta + 0.05 * (space.high - space.low) * rng.standard_normal((32, space.dim))
    # Reflect starting points that fall outside the ranges back inside (clipping
    # would stack walkers on the boundary, where emcee cannot separate them).
    start = space.low + np.abs(start - space.low)
    start = space.high - np.abs(space.high - start)

    for name, error in likelihoods.items():
        like = GaussianLikelihood(emu, data, NOISE * truth, emulator_error=error)
        sampler = emcee.EnsembleSampler(32, space.dim, like, vectorize=True)
        sampler.run_mcmc(start, 4000, progress=False)
        chain = sampler.get_chain(discard=1500, flat=True)
        lo, hi = np.percentile(chain, [15.87, 84.13], axis=0)
        hits[name].append((theta >= lo) & (theta <= hi))
    print(f"realization {r + 1}/{N_REALIZATIONS} done")

print("\nHow often the 68% interval contains the true value (ideal: 68%)")
coverage = {}
for name, h in hits.items():
    h = np.array(h)
    coverage[name] = h.mean()
    per_param = ", ".join(f"{n} {100 * c:.0f}%" for n, c in zip(space.names, h.mean(axis=0)))
    print(f"  {name:<28} {100 * h.mean():3.0f}%   ({per_param})")

fig, ax = plt.subplots(figsize=(7, 3.8))
names = list(coverage)
ax.barh(names, [100 * coverage[n] for n in names], color=["#c0392b", "#e67e22", "#1f4e79"])
ax.axvline(68.3, color="k", ls="--", lw=1.2, label="honest error bars (68%)")
for i, n in enumerate(names):
    ax.text(100 * coverage[n] + 1, i, f"{100 * coverage[n]:.0f}%", va="center")
ax.set_xlim(0, 100)
ax.invert_yaxis()
ax.set_xlabel("how often the 68% interval contains the true parameter (%)")
ax.set_title(f"Coverage over {N_REALIZATIONS} simulated observations")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("docs/images/coverage.png", dpi=130)
print("\nSaved docs/images/coverage.png")
