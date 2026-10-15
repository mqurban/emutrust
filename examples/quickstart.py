"""emutrust in one minute: from a slow model to honest parameter estimates.

Uses the built-in toy model so it runs anywhere in about a minute. Swap
``toy_spectrum`` and ``TOY_SPACE`` for your own slow model and parameter
ranges to use it for real.

    pip install "emutrust[mcmc]"
    python quickstart.py
"""

import emcee
import numpy as np

from emutrust import Emulator, GaussianLikelihood, ParameterSpace, run_model, validate
from emutrust.toy import TOY_SPACE, toy_spectrum

# 1. Define the parameters and their ranges, and choose training points.
space = ParameterSpace(TOY_SPACE)
X_train = space.sample(200, seed=0)  # Latin hypercube
X_valid = space.sample(100, seed=1)

# 2. Run the slow model at every point (checkpointed, so a crash loses nothing).
training = run_model(toy_spectrum, X_train, space, checkpoint="train_runs.npz", progress=False)
validation = run_model(toy_spectrum, X_valid, space, checkpoint="valid_runs.npz", progress=False)

# 3. Train the emulator: an ensemble of small neural networks.
emu = Emulator.fit(training, n_members=3, hidden=(64, 64), epochs=500, seed=0)

# 4. Check it on runs it has never seen.
report = validate(emu, validation)
print(report.summary())

# 5. Simulate an observation with known true parameters and 1% noise.
truth = np.array([1.2, -1.1, 3.0])
model = toy_spectrum(space.to_dict(truth))
rng = np.random.default_rng(42)
data = model * (1 + 0.01 * rng.standard_normal(model.size))

# 6. Infer the parameters with a likelihood that includes the emulator's error.
like = GaussianLikelihood(emu, data, noise_std=0.01 * model, emulator_error=report)
start = truth + 0.01 * (space.high - space.low) * rng.standard_normal((32, space.dim))
sampler = emcee.EnsembleSampler(32, space.dim, like, vectorize=True)
sampler.run_mcmc(start, 2000, progress=False)
chain = sampler.get_chain(discard=500, flat=True)

print("\nparameter      true   estimate (68% interval)")
for name, t, column in zip(space.names, truth, chain.T):
    lo, mid, hi = np.percentile(column, [15.87, 50, 84.13])
    print(f"{name:>9}  {t:8.3f}   {mid:.3f}  [{lo:.3f}, {hi:.3f}]")
print("\nA 68% interval should contain the true value about 2 times in 3, so one")
print("parameter landing just outside it is expected, not a failure.")
