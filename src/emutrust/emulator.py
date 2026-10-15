"""Neural network emulators with ensemble uncertainty."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn

from .runner import TrainingSet

PathLike = Union[str, Path]


class Standardizer:
    """Shift and scale each column to mean 0 and standard deviation 1.

    Neural networks train best when every input and output has a similar
    size; without this, a parameter like H0 (~70) would swamp one like
    omch2 (~0.12).
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)

    @classmethod
    def fit(cls, data: np.ndarray) -> Standardizer:
        return cls(data.mean(axis=0), data.std(axis=0))

    def transform(self, data: np.ndarray) -> np.ndarray:
        # A column that never changes has std 0; it maps to 0 instead of
        # dividing by zero.
        return (data - self.mean) / np.where(self.std > 0, self.std, 1.0)

    def inverse(self, data: np.ndarray) -> np.ndarray:
        # Multiplying by std 0 returns a constant column exactly, whatever
        # the network outputs for it.
        return data * self.std + self.mean


def _should_use_log(Y: np.ndarray) -> bool:
    """Use log outputs when every output is positive and spans >2 decades."""
    if np.any(Y <= 0):
        return False
    return bool(np.max(np.log10(Y.max(axis=0) / Y.min(axis=0))) > 2.0)


def _make_network(n_in: int, n_out: int, hidden: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.SiLU()]
        width = h
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


class Emulator:
    """A fast stand-in for a slow model, built from an ensemble of networks.

    Several networks are trained on the same training set, each starting
    from different random weights and seeing a different random split of
    the data. Where the training data pin the answer down, they agree;
    where data are thin or the function is hard to learn, they disagree.
    The spread of their predictions is the emulator's estimate of its own
    error.

    Example
    -------
    >>> emu = Emulator.fit(training, n_members=5, seed=0)
    >>> mean, std = emu.predict({"H0": 67.0, "omch2": 0.12, "ns": 0.96})
    """

    def __init__(
        self,
        members: list[nn.Sequential],
        x_scaler: Standardizer,
        y_scaler: Standardizer,
        log_outputs: bool,
        names: tuple[str, ...],
        config: dict,
    ):
        self.members = members
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.log_outputs = log_outputs
        self.names = tuple(names)
        self.config = config
        for net in self.members:
            net.eval()

    @property
    def n_members(self) -> int:
        return len(self.members)

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        training: TrainingSet,
        n_members: int = 5,
        hidden: Sequence[int] = (128, 128, 128),
        epochs: int = 2000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        val_fraction: float = 0.1,
        patience: int = 200,
        log_outputs: bool | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> Emulator:
        """Train an ensemble of ``n_members`` networks on a training set.

        ``log_outputs`` controls whether the networks learn ``log10`` of the
        outputs. ``None`` (the default) decides automatically: log is used
        when every output is positive and spans more than two orders of
        magnitude, as power spectra do.

        Each member trains until its error on its own held-out
        ``val_fraction`` of the data has not improved for ``patience``
        epochs, and keeps the weights from its best epoch ("early stopping",
        which prevents memorising the training points).
        """
        X = np.asarray(training.X, dtype=float)
        Y = np.asarray(training.Y, dtype=float)
        if n_members < 1:
            raise ValueError("n_members must be at least 1")
        if len(X) < 10:
            raise ValueError(f"Need at least 10 training points, got {len(X)}")
        if not 0 < val_fraction < 0.5:
            raise ValueError("val_fraction must be between 0 and 0.5")

        if log_outputs is None:
            log_outputs = _should_use_log(Y)
        elif log_outputs and np.any(Y <= 0):
            raise ValueError("log_outputs=True needs strictly positive outputs")
        Yt = np.log10(Y) if log_outputs else Y

        x_scaler = Standardizer.fit(X)
        y_scaler = Standardizer.fit(Yt)
        Xs = torch.tensor(x_scaler.transform(X), dtype=torch.float32)
        Ys = torch.tensor(y_scaler.transform(Yt), dtype=torch.float32)

        rng = np.random.default_rng(seed)
        members = []
        for m in range(n_members):
            member_seed = int(rng.integers(2**31))
            net, best = _train_member(
                Xs,
                Ys,
                hidden,
                epochs,
                batch_size,
                learning_rate,
                val_fraction,
                patience,
                member_seed,
            )
            if verbose:
                print(f"  member {m + 1}/{n_members}: best validation loss {best:.2e}")
            members.append(net)

        config = {
            "hidden": list(hidden),
            "n_inputs": X.shape[1],
            "n_outputs": Y.shape[1],
            "n_train": len(X),
        }
        return cls(members, x_scaler, y_scaler, bool(log_outputs), training.names, config)

    # -------------------------------------------------------------- predict
    def _as_array(self, x) -> np.ndarray:
        if isinstance(x, Mapping):
            missing = set(self.names) - set(x)
            if missing:
                raise ValueError(f"Missing parameters: {sorted(missing)}")
            x = [x[n] for n in self.names]
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != len(self.names):
            raise ValueError(f"Expected {len(self.names)} parameters, got {x.shape[1]}")
        return x

    def predict_members(self, x) -> np.ndarray:
        """Each member's prediction, shape ``(n_members, n_points, n_outputs)``."""
        xs = torch.tensor(self.x_scaler.transform(self._as_array(x)), dtype=torch.float32)
        with torch.no_grad():
            out = np.stack([net(xs).numpy() for net in self.members]).astype(float)
        out = self.y_scaler.inverse(out)
        return 10.0**out if self.log_outputs else out

    def predict(self, x, return_std: bool = True):
        """Predict outputs at one point (a dict or a 1-D array) or many points.

        Returns ``(mean, std)``: the ensemble average and the spread between
        members, both in the model's original output units. A single input
        point gives 1-D arrays; many points give 2-D arrays.
        """
        single = isinstance(x, Mapping) or np.ndim(x) == 1
        preds = self.predict_members(x)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0, ddof=1) if self.n_members > 1 else np.zeros_like(mean)
        if single:
            mean, std = mean[0], std[0]
        return (mean, std) if return_std else mean

    def __call__(self, x):
        return self.predict(x, return_std=False)

    # ----------------------------------------------------------- save/load
    def save(self, path: PathLike) -> None:
        """Save the emulator to a single file."""
        torch.save(
            {
                "members": [net.state_dict() for net in self.members],
                "x_scaler": (self.x_scaler.mean, self.x_scaler.std),
                "y_scaler": (self.y_scaler.mean, self.y_scaler.std),
                "log_outputs": self.log_outputs,
                "names": self.names,
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load(cls, path: PathLike) -> Emulator:
        """Load an emulator saved with :meth:`save`."""
        state = torch.load(path, weights_only=False)
        cfg = state["config"]
        members = []
        for sd in state["members"]:
            net = _make_network(cfg["n_inputs"], cfg["n_outputs"], cfg["hidden"])
            net.load_state_dict(sd)
            members.append(net)
        return cls(
            members,
            Standardizer(*state["x_scaler"]),
            Standardizer(*state["y_scaler"]),
            state["log_outputs"],
            tuple(state["names"]),
            cfg,
        )


def _train_member(Xs, Ys, hidden, epochs, batch_size, lr, val_fraction, patience, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    n = len(Xs)
    perm = torch.randperm(n, generator=gen)
    n_val = max(1, int(round(val_fraction * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    Xtr, Ytr, Xval, Yval = Xs[train_idx], Ys[train_idx], Xs[val_idx], Ys[val_idx]

    net = _make_network(Xs.shape[1], Ys.shape[1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=max(patience // 4, 1)
    )
    loss_fn = nn.MSELoss()

    best_loss, best_state, since_best = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(Xtr), generator=gen)
        for start in range(0, len(Xtr), batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(Xval), Yval).item()
        sched.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_state, since_best = val_loss, copy.deepcopy(net.state_dict()), 0
        else:
            since_best += 1
            if since_best >= patience:
                break

    net.load_state_dict(best_state)
    net.eval()
    return net, best_loss


__all__ = ["Emulator", "Standardizer"]
