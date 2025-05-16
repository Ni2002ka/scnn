from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scnn.optimize import optimize
from scnn.regularizers import NeuronGL1


def mse(y_pred, y):
    return np.mean(np.square(y_pred - y))


def unscale_log_return(scale_min: float, scale_max: float, scaled_log_rets: float) -> float:
    """
    Inverse scaling function to recover the original log returns from the scaled log returns.
    """
    # Recover the original log returns
    return (scaled_log_rets * (scale_max - scale_min)) + scale_min


def prepare_from_csv(
    clean_csv: str | Path,
    p: int = 10,
    split_date: str = "2018-01-02",
    prediction_cols: list[str] = ["log_ret", "sigma20"],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray, float, float]]]:
    """
    Load the *clean* CSV, build lag features + volatility column, then
    chronologically split into train / test blocks.

    Returns
    -------
    X_train, y_train : np.ndarray
        Input and primary target for training.

    output_dict : Dict[str, Tuple[np.ndarray, np.ndarray, float, float]]
        For each prediction column, includes (y_train, y_test, min, max)
    """
    df = (
        pd.read_csv(clean_csv, header=[0, 1], index_col=0, parse_dates=True)
        .rename(columns=str.strip)
    )

    train_dict = {}
    # Min-max scale
    def scale_x(x):
        return float(x.min()), float(x.max()), (x - x.min()) / (x.max() - x.min())

    col_stats = {}
    for col in prediction_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
        col_min, col_max, df[col] = scale_x(df[col])
        col_stats[col] = (col_min, col_max)


    # Subset to just the features we care about
    df = df[prediction_cols].dropna()

    # Find the split point in the date-indexed dataframe
    split_index = df.index.get_loc(pd.to_datetime(split_date))

    # Construct lagged feature/target set
    X = []
    target_data = {col: [] for col in prediction_cols}

    for i in range(p, len(df)):
        X.append(df.iloc[i - p:i].values.flatten())
        for col in prediction_cols:
            target_data[col].append(df.iloc[i][col])

    X = np.array(X)
    targets_dict = {}

    # Adjust for lag offset in split index
    relative_split_idx = split_index - p

    X_train, X_test = X[:relative_split_idx], X[relative_split_idx:]
    for col in prediction_cols:
        y_full = np.array(target_data[col])
        y_train = y_full[:relative_split_idx]
        y_test = y_full[relative_split_idx:]
        col_min, col_max = col_stats[col]
        targets_dict[col] = (y_train, y_test, col_min, col_max)

    return X_train, X_test, targets_dict


def run_model_single_col(X_tr, X_te, pred_tuple, max_neurons=256, lam_gl1=5e-4, huber_delta=0.1,
              verbose=False, plot=False, col_name="log_ret"):
    y_tr, y_te, scale_min, scale_max = pred_tuple
    # Run model for each prediction column
    model, _ = optimize(
        formulation="gated_relu",
        max_neurons=max_neurons,
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        bias=True,
        loss_type="huber",
        huber_delta=huber_delta,
        regularizer=NeuronGL1(lam_gl1),
        verbose=verbose,
        device="cpu",
    )

    # evaluation
    preds = model(X_te)
    preds_train = model(X_tr)

    test_error = mse(preds, y_te)
    train_error = mse(preds_train, y_tr)

    if plot:
        # Unscale the predictions and true values
        preds = unscale_log_return(scale_min, scale_max, preds.ravel())
        y_te = unscale_log_return(scale_min, scale_max, y_te.ravel())
        plt.figure(figsize=(10, 5))
        plt.plot(y_te, label='True Values')
        plt.plot(preds, label='Predictions')
        plt.title(f"Test Predictions vs True Values (MSE: {test_error:.4f}), {col_name}")
        plt.legend()
        plt.show()

        # Plot train predictions
        preds_train = unscale_log_return(scale_min, scale_max, preds_train.ravel())
        y_tr = unscale_log_return(scale_min, scale_max, y_tr.ravel())
        plt.figure(figsize=(10, 5))
        plt.plot(y_tr, label='True Values')
        plt.plot(preds_train, label='Predictions')
        plt.title(f"Training Predictions vs True Values (MSE: {train_error:.4f}), {col_name}")
        plt.legend()
        plt.show()

    return train_error, test_error


def run_model(clean_csv, p=10, max_neurons=256, lam_gl1=5e-4, huber_delta=0.1, prediction_cols=["log_ret", "sigma20"] ,verbose=False, plot=False):

    X_tr, X_te, target_dict = prepare_from_csv(clean_csv, p=p)

    # Run model for each prediction column
    train_error = {}
    test_error = {}
    for col, pred_tuple in target_dict.items():
        train_error[col], test_error[col] = run_model_single_col(
            X_tr, X_te, pred_tuple, max_neurons=max_neurons,
            lam_gl1=lam_gl1, huber_delta=huber_delta,
            verbose=verbose, plot=plot, col_name=col
        )

    return train_error, test_error


# nika's code modifed to work with my data
def main() -> None:
    # paths / parameters
    clean_csv   = "sp500_2005_2021_clean.csv"
    # best_hyperparams, best_test_mse = optimal_hyperparam_sweep(clean_csv)
    # lam_gl1_opt, huber_delta_opt, max_neurons_opt, p_opt = best_hyperparams
    # print(f"Best Hyperparameters: {best_hyperparams}") #(0.001, 0.1, 512, 32)
    # print(f"Best Test MSE: {best_test_mse}")
    lam_gl1_opt, huber_delta_opt, max_neurons_opt, p_opt = (0.001, 0.1, 512, 32)
    # run the model with optimal hyperparameters
    train_mse, test_mse = run_model(
        clean_csv,
        p=p_opt,
        max_neurons=max_neurons_opt,
        lam_gl1=lam_gl1_opt,
        huber_delta=huber_delta_opt,
        plot=True,
    )

if __name__ == "__main__":
    main()
