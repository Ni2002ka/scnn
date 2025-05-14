from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from scnn.optimize import optimize
from scnn.regularizers import NeuronGL1


def create_train_test_set(data: pd.DataFrame, p: int):
    X = []
    y = []

    log_rets = data['log_ret'].values

    for i in range(p, len(data)):

        block = data.iloc[i - p:i].values.flatten()  # shape: (p * num_features,)
        target = log_rets[i]
        X.append(block)
        y.append(target)

    return np.array(X), np.array(y)


def standardize_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # avoid divide-by-zero

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    return X_train_std.astype(np.float32), X_test_std.astype(np.float32)


def prepare_from_csv(
    clean_csv: str | Path,
    p: int = 10,
    vol_col: str = "sigma20",
    split_date: str = "2018-01-02",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the *clean* CSV, build lag features + volatility column, then
    chronologically split into train / test blocks.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        Shapes  (N_train, d) , (N_train, 1) , (N_test, d) , (N_test, 1)
    """
    df = (
        pd.read_csv(clean_csv, header=[0, 1], index_col=0, parse_dates=True)
        .rename(columns=str.strip)
    )

    df['log_ret'] *= 1000
    df['sigma20'] *= 10

    split_index = df.index.get_loc(pd.to_datetime(split_date))
    df = df[['log_ret', 'sigma20']].dropna()

    # Make sure to offset by `p` since X, y only start from `p`
    X, y = create_train_test_set(df, p)
    relative_split_idx = split_index - p

    X_train, X_test = X[:relative_split_idx], X[relative_split_idx:]
    y_train, y_test = y[:relative_split_idx], y[relative_split_idx:]
    # X_train_std, X_test_std = standardize_features(X_train, X_test)
    X_train_std, X_test_std = X_train, X_test

    return (
        X_train_std,
        y_train,
        X_test_std,
        y_test,
    )



# nika's code modifed to work with my data
def main() -> None:
    # paths / parameters
    clean_csv   = "sp500_2005_2021_clean.csv"
    p           = 10
    max_neurons = 1000
    lam_gl1     = 1e-4
    # lam_gl1     = 0.0
    huber_delta = .9

    # data!!
    X_tr, y_tr, X_te, y_te = prepare_from_csv(clean_csv, p=p)

    # model stuff
    model, _ = optimize(
        formulation="gated_relu",
        max_neurons=max_neurons,
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        loss_type="huber",
        huber_delta=huber_delta,
        regularizer=NeuronGL1(lam_gl1),
        verbose=True,
        device="cpu",
    )

    # evaluation
    preds     = model(X_te)
    print("Preds (first 5):", preds[:5].ravel())
    print("True  (first 5):", y_te[:5].ravel())

    preds_train = model(X_tr)
    print("Train preds (first 5):", preds_train[:5].ravel())
    print("Train true  (first 5):", y_tr[:5].ravel())

    pred_error = mse(preds, y_te)
    print(f"\nMSE : {pred_error:.8f}")
    print(f"Train samples      : {len(X_tr):,}")
    print(f"Test  samples      : {len(X_te):,}")
    print(f"Feature dimension  : {X_tr.shape[1]}")

def mse(y_pred, y):
    return np.mean(np.square(y_pred - y))

if __name__ == "__main__":
    main()
