"""
Train linear + neural surrogates on the generated dataset and report metrics.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.surrogate.models import (
    FEATURE_NAMES,
    TARGET,
    LinearSurrogate,
    NeuralSurrogate,
    DeepEnsemble,
    regression_metrics,
)
from src.solver.dataset import split_dataset


def main():
    path = "data/demo_dataset.csv"
    if not os.path.exists(path):
        from src.solver.dataset import generate_dataset
        generate_dataset(n_samples=80, n_workers=2, out_path=path)

    df = pd.read_csv(path)
    train, val, test = split_dataset(df, seed=1)

    X_train = train[FEATURE_NAMES].values
    y_train = train[TARGET].values
    X_val = val[FEATURE_NAMES].values
    y_val = val[TARGET].values
    X_test = test[FEATURE_NAMES].values
    y_test = test[TARGET].values

    print(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")

    # Linear baseline
    lin = LinearSurrogate().fit(X_train, y_train)
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        metrics = regression_metrics(y, lin.predict(X))
        print(f"Linear  {name}: R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.0f}  "
              f"max_rel={metrics['max_rel_error']*100:.1f}%")

    # Neural net
    nn = NeuralSurrogate(n_hidden=32, n_layers=2, epochs=400, batch_size=16)
    nn.fit(X_train, y_train)
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        metrics = regression_metrics(y, nn.predict(X))
        print(f"NN      {name}: R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.0f}  "
              f"max_rel={metrics['max_rel_error']*100:.1f}%")

    # Quick ensemble uncertainty demo
    ens = DeepEnsemble(n_models=3, n_hidden=32, n_layers=2, epochs=200, batch_size=16)
    ens.fit(X_train, y_train)
    mean, std = ens.predict(X_test)
    print(f"Ensemble test  mean max_rel error = "
          f"{np.max(np.abs(mean-y_test)/(np.abs(y_test)+1e-12))*100:.1f}%  "
          f"avg predictive std = {std.mean():.0f}")


if __name__ == "__main__":
    main()
