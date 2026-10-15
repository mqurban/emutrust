"""Show how an ensemble's spread grows where it has no training data.

Run from the repository root:

    python examples/ensemble_uncertainty.py
"""

import matplotlib.pyplot as plt
import numpy as np

from emutrust import Emulator, TrainingSet


def true_model(x):
    return np.sin(6 * x) + x


x_train = np.linspace(0.0, 0.6, 40)[:, None]
training = TrainingSet(X=x_train, Y=true_model(x_train), names=("x",), seconds_per_run=0.0)
emu = Emulator.fit(training, n_members=5, hidden=(32, 32), epochs=800, patience=150, seed=1)

x = np.linspace(0.0, 1.0, 300)[:, None]
members = emu.predict_members(x)[:, :, 0]
mean, std = emu.predict(x)
mean, std = mean[:, 0], std[:, 0]

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axvspan(0.6, 1.0, color="#f2f2f2", label="no training data")
for i, m in enumerate(members):
    ax.plot(x, m, color="#7aa6c2", lw=1, alpha=0.8, label="ensemble members" if i == 0 else None)
ax.fill_between(
    x[:, 0], mean - 2 * std, mean + 2 * std, color="#1f4e79", alpha=0.15, label="mean ± 2 std"
)
ax.plot(x, true_model(x), "k--", lw=1.5, label="true model")
ax.plot(x_train, training.Y, "o", color="#c0392b", ms=4, label="training runs")
ax.set_xlabel("parameter")
ax.set_ylabel("model output")
ax.set_title(
    "Inside the data the members agree; outside it they spread apart, but can all be wrong"
)
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig("docs/images/ensemble_uncertainty.png", dpi=130)
print("Saved docs/images/ensemble_uncertainty.png")
