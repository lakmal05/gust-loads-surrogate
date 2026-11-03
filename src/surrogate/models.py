"""
Surrogate models for peak root-bending-moment prediction.

Baselines:
  - Linear regression (closed-form)
  - Small fully-connected neural network (PyTorch)
  - Optional deep ensemble for uncertainty
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_NAMES = ["H", "U_ds", "V", "rho", "mass_factor"]
TARGET = "peak_rbm"


class LinearSurrogate:
    """Ordinary least-squares with optional intercept."""

    def __init__(self):
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.mu_: Optional[np.ndarray] = None
        self.sigma_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSurrogate":
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0) + 1e-12
        Xn = (X - self.mu_) / self.sigma_
        # design matrix with intercept
        A = np.column_stack([np.ones(len(Xn)), Xn])
        theta, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.mu_) / self.sigma_
        return self.intercept_ + Xn @ self.coef_


class MLP(nn.Module):
    def __init__(self, n_in: int = 5, n_hidden: int = 64, n_layers: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [n_in] + [n_hidden] * n_layers + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralSurrogate:
    """Small feed-forward network trained with Adam + MSE."""

    def __init__(
        self,
        n_hidden: int = 64,
        n_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 300,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model: Optional[MLP] = None
        self.mu_: Optional[np.ndarray] = None
        self.sigma_: Optional[np.ndarray] = None
        self.y_mu_: float = 0.0
        self.y_sigma_: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralSurrogate":
        self.mu_ = X.mean(0)
        self.sigma_ = X.std(0) + 1e-12
        self.y_mu_ = float(y.mean())
        self.y_sigma_ = float(y.std()) + 1e-12

        Xn = (X - self.mu_) / self.sigma_
        yn = (y - self.y_mu_) / self.y_sigma_

        Xt = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        yt = torch.tensor(yn, dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        self.model = MLP(X.shape[1], self.n_hidden, self.n_layers).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for ep in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        Xn = (X - self.mu_) / self.sigma_
        Xt = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            yn = self.model(Xt).cpu().numpy()
        return yn * self.y_sigma_ + self.y_mu_


class DeepEnsemble:
    """Ensemble of neural networks for epistemic uncertainty."""

    def __init__(self, n_models: int = 5, **nn_kwargs):
        self.n_models = n_models
        self.models: List[NeuralSurrogate] = [
            NeuralSurrogate(**nn_kwargs) for _ in range(n_models)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepEnsemble":
        for i, m in enumerate(self.models):
            # different random initialisation is enough for diversity
            m.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns mean and std across the ensemble."""
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return preds.mean(0), preds.std(0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    max_rel = float(np.max(np.abs(err) / (np.abs(y_true) + 1e-12)))
    return {"rmse": rmse, "mae": mae, "r2": r2, "max_rel_error": max_rel}
