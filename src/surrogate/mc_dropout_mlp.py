"""The one surrogate architecture used across every representation
(paper draft Sec 3.2): d -> 512 -> 128 -> 1, ReLU, dropout p=0.2, trained
with MSE, uncertainty from T=50 stochastic forward passes at inference
(dropout left active). Deliberately a single fixed architecture -- see
DESIGN.md Sec 4.3 for why this project does not use ALSU's dual-MVE-head
+ Spearman-loss surrogate: representation comparisons must go through an
identical model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(512, 128), dropout: float = 0.2):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class McDropoutMLP:
    """fit(X, y) / predict(X) -> (mu, sigma), matching the paper's
    MC-Dropout surrogate exactly. Not the molpal `Model` ABC itself --
    see model_adapter.py for the bridge that plugs this into Explorer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims=(512, 128),
        dropout: float = 0.2,
        mc_samples: int = 50,
        lr: float = 3e-4,
        epochs: int = 50,
        batch_size: int = 256,
    ):
        self.mc_samples = mc_samples
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._net = _MLP(in_dim, hidden_dims, dropout).to(DEVICE)
        self._ym = self._ys = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._ym = float(y.mean())
        self._ys = float(y.std()) + 1e-8
        y_norm = (y - self._ym) / self._ys

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y_norm, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True
        )
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self._net.train()  # dropout active during training (standard)
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(self._net(xb), yb)
                loss.backward()
                opt.step()

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """T stochastic forward passes with dropout left active
        (MC Dropout, Gal & Ghahramani 2016): mu = mean, sigma = std.
        """
        Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        self._net.train()  # keep dropout active at inference time

        preds = torch.empty(self.mc_samples, X.shape[0])
        with torch.no_grad():
            for t in range(self.mc_samples):
                preds[t] = self._net(Xt).cpu()

        mu_norm = preds.mean(dim=0).numpy()
        sigma_norm = preds.std(dim=0).numpy()

        mu = mu_norm * self._ys + self._ym
        sigma = sigma_norm * self._ys  # scale-only; no shift for a spread stat
        return mu.astype(np.float32), sigma.astype(np.float32)
