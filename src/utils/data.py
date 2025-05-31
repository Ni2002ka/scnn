from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
import numpy as np


def mse(y_pred, y):
    return np.mean(np.square(y_pred - y))


def unscale_log_return(scale_min: float, scale_max: float, scaled_log_rets: float) -> float:
    """
    Inverse scaling function to recover the original log returns from the scaled log returns.
    """
    # Recover the original log returns
    return (scaled_log_rets * (scale_max - scale_min)) + scale_min


def prepare_single_asset_from_csv(
    clean_csv: str | Path,
    p: int = 10,
    split_date: str = "2018-01-02",
    prediction_cols: list[str] = ["log_ret", "sigma20", "Volume"]
) -> Dict[str, np.ndarray | Tuple[np.ndarray, np.ndarray, float, float]]:
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
    def scale_x(x, split_idx=None):
        if split_idx is None:
            split_idx = len(x)

        x_min = x.iloc[:split_idx].min()
        x_max = x.iloc[:split_idx].max()
        return float(x_min), float(x_max), (x - x_min) / (x_max - x_min)

    # Find the split point in the date-indexed dataframe
    split_index = df.index.get_loc(pd.to_datetime(split_date))

    col_stats = {}
    for col in prediction_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
        col_min, col_max, df[col] = scale_x(df[col], split_idx=split_index)
        col_stats[col] = (col_min, col_max)


    # Subset to just the features we care about
    df = df[prediction_cols].dropna()

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
    train_dict["X_train"] = X_train
    train_dict["X_test"] = X_test
    for col in prediction_cols:
        y_full = np.array(target_data[col])
        y_train = y_full[:relative_split_idx]
        y_test = y_full[relative_split_idx:]
        col_min, col_max = col_stats[col]
        targets_dict[col] = (y_train, y_test, col_min, col_max)

    train_dict["targets"] = targets_dict
    return train_dict


def prepare_from_csv(
    ticker_list: list[str],
    p: int = 10,
    split_date: str = "2018-01-02",
    prediction_cols: list[str] = ["log_ret", "sigma20", "Volume"],
    asset_correlation: bool = True
) -> Dict[str, Dict[str, np.ndarray | Tuple[np.ndarray, np.ndarray, float, float]]]:
    """
    Load the *clean* CSVs, build lag features + volatility column, then
    chronologically split into train / test blocks.

    asset_correlation: bool: If True, will concatenate all assets into a single X matrix.
    This can leverage correlations between assets, but may not be suitable for all tasks.

    Returns
    -------
    X_train, y_train : np.ndarray
        Input and primary target for training.

    output_dict : Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, float, float]]]
        Dictionary where keys are tickers, and values are dictionaries for each ticker.
        For each prediction column, includes (y_train, y_test, min, max)
    """

    all_X_tr = []
    all_X_te = []
    target_dict = {}
    for ticker in ticker_list:
        clean_csv = f"{ticker}_2005_2021_clean.csv"
        ticker_dict = prepare_single_asset_from_csv(
            clean_csv,
            p=p,
            split_date=split_date,
            prediction_cols=prediction_cols
        )
        all_X_tr.append(ticker_dict["X_train"])
        all_X_te.append(ticker_dict["X_test"])

        target_dict[ticker] = ticker_dict

    if asset_correlation:
        # Concatenate all assets into a single X matrix
        all_X_tr = np.concatenate(all_X_tr, axis=1)
        all_X_te = np.concatenate(all_X_te, axis=1)

        for ticker in ticker_list:
            target_dict[ticker]["all_X_train"] = all_X_tr
            target_dict[ticker]["all_X_test"] = all_X_te

    return target_dict

