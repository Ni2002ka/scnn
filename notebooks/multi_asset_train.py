from __future__ import annotations
import time

import numpy as np
import matplotlib.pyplot as plt

from scnn.optimize import optimize
from scnn.regularizers import NeuronGL1

from utils.data import prepare_from_csv, unscale_log_return, mse
from non_convex_models.LSTM import train_lstm
from non_convex_models.transformer import train_transformer
from non_convex_models.mlp import train_mlp
from utils.PCA import find_critical_indices_map


def train_single_col(X_tr, X_te, pred_tuple, model_type='scnn', max_neurons=256, lam_gl1=5e-4, huber_delta=0.1,
                     verbose=False, p=10, PCA_operator=None):
    y_tr, y_te, scale_min, scale_max = pred_tuple
    if model_type == 'scnn':
        X_tr_reshaped = X_tr.reshape(X_tr.shape[0], -1)
        X_te_reshaped = X_te.reshape(X_te.shape[0], -1)
        if PCA_operator is not None:
            scaler, pca = PCA_operator
            X_tr_reshaped = pca.transform(scaler.transform(X_tr_reshaped))
            X_te_reshaped = pca.transform(scaler.transform(X_te_reshaped))
        return train_scnn(X_tr_reshaped, X_te_reshaped, pred_tuple, max_neurons=max_neurons, lam_gl1=lam_gl1,
                          huber_delta=huber_delta, verbose=verbose)
    elif model_type == 'lstm':
        return train_lstm(X_tr, y_tr, X_te, y_te, scale_min, scale_max, huber_delta=huber_delta)
    elif model_type == 'transformer':
        return train_transformer(X_tr, y_tr, X_te, y_te, scale_min, scale_max, huber_delta=huber_delta)
    elif model_type == 'mlp':
        X_tr_reshaped = X_tr.reshape(X_tr.shape[0], -1)
        X_te_reshaped = X_te.reshape(X_te.shape[0], -1)
        return train_mlp(X_tr_reshaped, y_tr, X_te_reshaped, y_te, scale_min, scale_max, huber_delta=huber_delta)


def train_scnn(X_tr, X_te, pred_tuple, max_neurons=256, lam_gl1=5e-4, huber_delta=0.1,
                     verbose=False):
    y_tr, y_te, scale_min, scale_max = pred_tuple

    # Run model
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

    # Predictions
    preds_te = model(X_te).ravel()
    preds_tr = model(X_tr).ravel()

    # Compute MSE
    test_error = mse(preds_te, y_te)
    train_error = mse(preds_tr, y_tr)

    # Unscale predictions + true
    preds_te_unscaled = unscale_log_return(scale_min, scale_max, preds_te)
    y_te_unscaled = unscale_log_return(scale_min, scale_max, y_te.ravel())
    preds_tr_unscaled = unscale_log_return(scale_min, scale_max, preds_tr)
    y_tr_unscaled = unscale_log_return(scale_min, scale_max, y_tr.ravel())

    return train_error, test_error, model, (y_tr_unscaled, preds_tr_unscaled), (y_te_unscaled, preds_te_unscaled)


def train_nets(ticker_list, p=10, max_neurons=256, lam_gl1=5e-4, huber_delta=0.1, model_type='scnn',
               prediction_cols=["log_ret", "sigma20", "Volume"], verbose=False, plot=False, asset_correlation=False, PCA_operator=None):
    train_target_dict = prepare_from_csv(ticker_list, prediction_cols=prediction_cols, p=p, asset_correlation=asset_correlation)

    for ticker, train_dict in train_target_dict.items():

        X_tr = train_dict["X_train"]
        X_te = train_dict["X_test"]

        targets = train_dict["targets"]
        fig, axs = None, None
        if plot:
            fig, axs = plt.subplots(len(prediction_cols), 2, figsize=(14, 4 * len(prediction_cols)))
            if len(prediction_cols) == 1:
                axs = np.expand_dims(axs, axis=0)

        for i, (col, pred_tuple) in enumerate(targets.items()):
            if col == "log_ret":
                lam_gl1, huber_delta = (1e-3, 0.4)
            elif col == "Volume":
                lam_gl1, huber_delta = (1e-8, 0.05)
            elif col == "sigma20":
                lam_gl1, huber_delta = (1e-8, 0.05)

            start_time = time.time()
            if asset_correlation:
                X_tr_all = train_dict["all_X_train"]
                X_te_all = train_dict["all_X_test"]

                # Re-train the model with all assets
                train_error, test_error, model, (y_tr, preds_tr), (y_te, preds_te) = train_single_col(
                    X_tr_all, X_te_all, pred_tuple,
                    max_neurons=max_neurons,
                    lam_gl1=lam_gl1,
                    p=p,
                    model_type=model_type,
                    huber_delta=huber_delta,
                    verbose=verbose,
                    PCA_operator=PCA_operator
                )
            else:
                train_error, test_error, model, (y_tr, preds_tr), (y_te, preds_te) = train_single_col(
                    X_tr, X_te, pred_tuple,
                    max_neurons=max_neurons,
                    p=p,
                    model_type=model_type,
                    lam_gl1=lam_gl1,
                    huber_delta=huber_delta,
                    verbose=verbose,
                    PCA_operator=PCA_operator
                )

            elapsed_time = time.time() - start_time
            if verbose:
                print(f"Training {col} for {ticker} took {elapsed_time:.2f} seconds.")

            if plot:
                axs[i, 0].plot(y_tr, label='True', linewidth=1)
                axs[i, 0].plot(preds_tr, label='Pred', linewidth=1)
                # if asset_correlation:
                #     axs[i, 0].plot(preds_tr_all, label='Pred (All Assets)', linewidth=1)
                axs[i, 0].set_title(f"{col} - Train (MSE: {train_error:.4f})")
                axs[i, 0].legend()

                axs[i, 1].plot(y_te, label='True', linewidth=1)
                axs[i, 1].plot(preds_te, label='Pred', linewidth=1)
                # if asset_correlation:
                #     axs[i, 1].plot(preds_te_all, label='Pred (All Assets)', linewidth=1)
                axs[i, 1].set_title(f"{col} - Test (MSE: {test_error:.4f})")
                axs[i, 1].legend()

        if plot:
            fig.suptitle(f"Predictions for {ticker}", fontsize=16)
            fig.tight_layout()
            fig.subplots_adjust(top=0.92)
            plt.show()


def main() -> None:
    # paths / parameters
    ticker_list = ["AAPL", "MSFT", "GOOG", "sp500"]
    lam_gl1_opt, huber_delta_opt, max_neurons_opt, p_opt = (1e-5, 0.1, 512, 32)
    crash_dict = prepare_from_csv(ticker_list=ticker_list, p=p_opt, split_date="2008-09-29",
                                  prediction_cols=["log_ret", "sigma20", "Volume"], asset_correlation=True)

    pca = find_critical_indices_map(crash_dict)
    train_nets(
        ticker_list,
        p=p_opt,
        # model_type='transformer',
        # model_type='mlp',
        model_type='scnn',
        # model_type='lstm',
        max_neurons=max_neurons_opt,
        lam_gl1=lam_gl1_opt,
        huber_delta=huber_delta_opt,
        plot=True,
        # PCA_operator=pca,
        # asset_correlation=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
