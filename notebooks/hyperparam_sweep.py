from multi_asset_train import train_single_col
from utils.data import prepare_from_csv


def run_model_per_col(ticker, col_name, p=10, max_neurons=256, lam_gl1=5e-4, huber_delta=0.1, model_type='scnn'):
    """
    Run the model for a specific column on the sp500 dataset.
    """
    train_target_dict = prepare_from_csv([ticker], p=p)
    train_dict = train_target_dict[ticker]
    X_tr = train_dict["X_train"]
    X_te = train_dict["X_test"]
    targets = train_dict["targets"]

    if col_name in targets:
        pred_tuple = targets[col_name]
        train_error, test_error, _, _, _ = train_single_col(
            X_tr, X_te, pred_tuple,
            model_type=model_type,
            max_neurons=max_neurons,
            lam_gl1=lam_gl1,
            huber_delta=huber_delta,
            p=p,
        )
        return train_error, test_error
    else:
        return None, None


def optimal_hyperparam_sweep(ticker="sp500", model_type='scnn'):
    lam_gl1_values = [1e-8, 1e-5, 1e-3]
    huber_delta_values = [0.01, 0.1, 1]
    hidden_neurons = [128, 256, 512]
    window_sizes = [8, 16, 32]

    prediction_cols = ["log_ret", "sigma20", "Volume"]
    best_hyperparams_per_col = {}

    for col_name in prediction_cols:
        best_hyperparams = None
        best_test_mse = float('inf')

        for lam_gl1 in lam_gl1_values:
            for huber_delta in huber_delta_values:
                for max_neurons in hidden_neurons:
                    for p in window_sizes:
                        train_mse, test_mse = run_model_per_col(
                            ticker=ticker,
                            col_name=col_name,
                            p=p,
                            max_neurons=max_neurons,
                            lam_gl1=lam_gl1,
                            huber_delta=huber_delta,
                            model_type=model_type,
                        )

                        if test_mse is None:
                            continue

                        # print(f"{col_name} | lam_gl1: {lam_gl1}, huber_delta: {huber_delta}, max_neurons: {max_neurons}, p: {p} --> Test MSE: {test_mse:.6f}")

                        if test_mse < best_test_mse:
                            best_test_mse = test_mse
                            best_hyperparams = (lam_gl1, huber_delta, max_neurons, p)

        best_hyperparams_per_col[col_name] = {
            "best_hyperparams": best_hyperparams,
            "best_test_mse": best_test_mse
        }

        print(f"\nBest for {col_name}: {best_hyperparams} | Test MSE: {best_test_mse:.6f}")

    return best_hyperparams_per_col


def main():
    results = optimal_hyperparam_sweep(
        ticker="sp500",
        model_type='scnn',
    )
    print("\nFinal Best Hyperparameters per Column:")
    for col, res in results.items():
        print(f"{col}: {res}")


if __name__ == "__main__":
    main()
