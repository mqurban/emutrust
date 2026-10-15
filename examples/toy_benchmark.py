"""Accuracy and uncertainty calibration of an emulator on the toy spectrum.

Run from the repository root (takes about a minute on a laptop CPU):

    python examples/toy_benchmark.py
"""

import numpy as np

from emutrust import Emulator, ParameterSpace, run_model
from emutrust.toy import TOY_SPACE, toy_spectrum

space = ParameterSpace(TOY_SPACE)
training = run_model(toy_spectrum, space.sample(300, seed=0), space, progress=False)
test = run_model(toy_spectrum, space.sample(200, method="random", seed=99), space, progress=False)

emu = Emulator.fit(training, n_members=5, seed=0)
mean, std = emu.predict(test.X)

frac_err = np.abs(mean - test.Y) / test.Y
pulls = np.abs(mean - test.Y) / std  # error measured in units of predicted std

print(f"Training runs: {len(training)}, test points: {len(test)}")
print(f"Mean fractional error:   {100 * frac_err.mean():.2f}%")
print(f"Median fractional error: {100 * np.median(frac_err):.2f}%")
print(f"Worst fractional error:  {100 * frac_err.max():.1f}%")
print(f"Truth within 1 std: {100 * np.mean(pulls < 1):.0f}%  (68% if calibrated)")
print(f"Truth within 2 std: {100 * np.mean(pulls < 2):.0f}%  (95% if calibrated)")
