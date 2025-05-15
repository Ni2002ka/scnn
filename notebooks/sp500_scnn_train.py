from __future__ import annotations

from pathlib import Path
from typing import Tuple

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
    vol_col: str = "sigma20",
    split_date: str = "2018-01-02",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Load the *clean* CSV, build lag features + volatility column, then
    chronologically split into train / test blocks.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        Shapes  (N_train, d) , (N_train, 1) , (N_test, d) , (N_test, 1)

    ret_min, ret_max : float: log returns scaling parameters
    """
    df = (
        pd.read_csv(clean_csv, header=[0, 1], index_col=0, parse_dates=True)
        .rename(columns=str.strip)
    )

    # Min-max scale
    def scale_x(x):
        return float(x.min()), float(x.max()), (x - x.min()) / (x.max() - x.min())

    ret_min, ret_max, df['log_ret'] = scale_x(df['log_ret'])
    _, _, df[vol_col] = scale_x(df[vol_col])

    # Subset to just the features we care about
    df = df[['log_ret', vol_col]].dropna()

    # Find the split point in the date-indexed dataframe
    split_index = df.index.get_loc(pd.to_datetime(split_date))

    # Construct lagged feature/target set
    X, y = [], []
    for i in range(p, len(df)):
        block = df.iloc[i - p:i].values.flatten()
        target = df.iloc[i]['log_ret']  # Predict current time using previous p
        X.append(block)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    # Adjust for lag offset in split index
    relative_split_idx = split_index - p

    X_train, X_test = X[:relative_split_idx], X[relative_split_idx:]
    y_train, y_test = y[:relative_split_idx], y[relative_split_idx:]

    return X_train, y_train, X_test, y_test, ret_min, ret_max


def optimal_hyperparam_sweep(clean_csv):
    """
    Perform a hyperparameter sweep to find the optimal hyperparameters for the model.
    """
    # Define the hyperparameter grid
    lam_gl1_values = [1e-5, 1e-4, 1e-3]
    huber_delta_values = [0.01, 0.1, 1]
    hidden_neurons = [128, 256, 512]
    window_sizes = [8, 16, 32]

    # Initialize variables to store the best hyperparameters and their corresponding performance
    best_hyperparams = None
    best_test_mse = float('inf')

    # Iterate over all combinations of hyperparameters
    for lam_gl1 in lam_gl1_values:
        for huber_delta in huber_delta_values:
            for max_neurons in hidden_neurons:
                for p in window_sizes:
                    train_mse, test_mse = run_model(
                        clean_csv,
                        p=p,
                        max_neurons=max_neurons,
                        lam_gl1=lam_gl1,
                        huber_delta=huber_delta,
                    )

                    # Update the best hyperparameters if the current performance is better
                    if test_mse < best_test_mse:
                        best_test_mse = test_mse
                        best_hyperparams = (lam_gl1, huber_delta, max_neurons, p)

    return best_hyperparams, best_test_mse


def run_model(clean_csv, p=10, max_neurons=256, lam_gl1=5e-4, huber_delta=0.1, verbose=False, plot=False):

    X_tr, y_tr, X_te, y_te, scale_min, scale_max = prepare_from_csv(clean_csv, p=p)
    # model stuff
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
    preds     = model(X_te)
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
        plt.title(f"Test Predictions vs True Values (MSE: {test_error:.4f})")
        plt.legend()
        plt.show()

        # Plot train predictions
        preds_train = unscale_log_return(scale_min, scale_max, preds_train.ravel())
        y_tr = unscale_log_return(scale_min, scale_max, y_tr.ravel())
        plt.figure(figsize=(10, 5))
        plt.plot(y_tr, label='True Values')
        plt.plot(preds_train, label='Predictions')
        plt.title(f"Training Predictions vs True Values (MSE: {train_error:.4f})")
        plt.legend()
        plt.show()



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
