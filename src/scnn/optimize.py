"""Convex-reformulation optimiser for SCNN models.

We added the following:
* loss_type == "pinball"  (quantile loss)
* pinball_tau float keyword for optimize / optimize_path
"""

from __future__ import annotations

import math
import os
import pickle as pkl
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
from typing_extensions import Literal

from scnn.activations import sample_gate_vectors
from scnn.metrics import Metrics
from scnn.models import ConvexGatedReLU, ConvexReLU, Model
from scnn.private.interface import (
    build_internal_model,
    build_internal_regularizer,
    build_metrics_tuple,
    build_optimizer,
    build_public_model,
    get_logger,
    normalized_into_input_space,
    process_data,
    set_device,
    update_public_metrics,
    update_public_model,
)
from scnn.private.models.solution_mappings import get_nc_formulation
from scnn.regularizers import Regularizer
from scnn.solvers import AL, RFISTA, Optimizer


Formulation = Literal["gated_relu", "relu"]
Device = Literal["cpu", "cuda"]
Dtype = Literal["float32", "float64"]



#   optimize (single fit)
def optimize(
    formulation: Formulation,
    max_neurons: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    *,
    regularizer: Optional[Regularizer] = None,
    loss_type: str = "least squares",
    huber_delta: float | None = None,
    pinball_tau: float | None = None,
    bias: bool = False,
    return_convex: bool = False,
    unitize_data: bool = True,
    verbose: bool = False,
    log_file: str | None = None,
    device: Device = "cpu",
    dtype: Dtype = "float32",
    seed: int = 778,
) -> Tuple[Model, Metrics]:
    """Train one convex SCNN model and return the fitted weights + metrics."""

    # Choose convex formulation + solver
    d = X_train.shape[1]
    c = 1 if y_train.ndim == 1 else y_train.shape[1]

    if formulation == "gated_relu":
        G = sample_gate_vectors(seed, d, max_neurons)
        model = ConvexGatedReLU(G, c=c, bias=bias)
        solver: Optimizer = RFISTA(model)
    elif formulation == "relu":
        G = sample_gate_vectors(seed, d, max_neurons // 2)
        model = ConvexReLU(G, c=c, bias=bias)
        solver = AL(model)
    else:  # pragma: no cover
        raise ValueError(f"Unknown formulation {formulation}")

    metrics = Metrics()

    return optimize_model(
        model=model,
        solver=solver,
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        loss_type=loss_type,
        huber_delta=huber_delta,
        pinball_tau=pinball_tau,
        regularizer=regularizer,
        return_convex=return_convex,
        unitize_data=unitize_data,
        verbose=verbose,
        log_file=log_file,
        device=device,
        dtype=dtype,
        seed=seed,
    )


 #  optimize_model (internal)

def optimize_model(
    model: Model,
    solver: Optimizer,
    metrics: Metrics,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    *,
    loss_type: str = "least squares",
    huber_delta: float | None = None,
    pinball_tau: float | None = None,
    regularizer: Optional[Regularizer] = None,
    return_convex: bool = False,
    unitize_data: bool = True,
    verbose: bool = False,
    log_file: str | None = None,
    device: Device = "cpu",
    dtype: Dtype = "float32",
    seed: int = 778,
) -> Tuple[Model, Metrics]:
    """Core solver routine (used by optimise & optimise_path)."""

    logger = get_logger("scnn", verbose, False, log_file)
    if solver.cpu_only() and device != "cpu":
        logger.warning("Solver forced to CPU; overriding device arg.")
        device = "cpu"
    set_device(device, dtype, seed)

    # Pre-processing (unitise columns etc.)
    (X_train, y_train), (X_test, y_test), col_norms = process_data(
        X_train, y_train, X_test, y_test, unitize_data, model.bias
    )

    internal_model = build_internal_model(
        model=model,
        regularizer=regularizer,
        X_train=X_train,
        loss_type=loss_type,
        huber_delta=huber_delta,
        pinball_tau=pinball_tau,
    )

    opt_proc = build_optimizer(solver, regularizer, metrics)
    _, internal_model, internal_metrics = opt_proc(
        logger,
        internal_model,
        lambda m: m,
        (X_train, y_train),
        (X_test, y_test),
        build_metrics_tuple(metrics),
    )
    metrics = update_public_metrics(metrics, internal_metrics)

    # Map weights back to original scale if we unitized
    if unitize_data:
        internal_model.weights = normalized_into_input_space(
            internal_model.weights, col_norms
        )

    if return_convex:
        return update_public_model(model, internal_model), metrics

    nc_internal = get_nc_formulation(internal_model, remove_sparse=True)
    public_model = build_public_model(nc_internal, model.bias)
    return public_model, metrics


def optimize_path(
    model: Model,
    solver: Optimizer,
    path: List[Regularizer],
    metrics: Metrics,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    *,
    loss_type: str = "least squares",
    huber_delta: float | None = None,
    pinball_tau: float | None = None,
    warm_start: bool = True,
    save_path: Optional[str] = None,
    return_convex: bool = False,
    unitize_data: bool = True,
    verbose: bool = False,
    log_file: str | None = None,
    device: Device = "cpu",
    dtype: Dtype = "float32",
    seed: int = 778,
) -> Tuple[List[Union[Model, str]], List[Metrics]]:
    """Solve the problem along a regularization path."""

    logger = get_logger("scnn", verbose, False, log_file)
    if solver.cpu_only() and device != "cpu":
        logger.warning("Solver forced to CPU; overriding device arg.")
        device = "cpu"
    set_device(device, dtype, seed)

    # preprocessing
    (X_train, y_train), (X_test, y_test), col_norms = process_data(
        X_train, y_train, X_test, y_test, unitize_data
    )

    internal_model = build_internal_model(
        model=model,
        regularizer=path[0],
        X_train=X_train,
        loss_type=loss_type,
        huber_delta=huber_delta,
        pinball_tau=pinball_tau,
    )
    initializer = lambda m: m

    metrics_list, model_list = [], []
    for reg in path:
        internal_model.regularizer = build_internal_regularizer(reg)
        opt_proc = build_optimizer(solver, reg, metrics)
        _, internal_model, int_metrics = opt_proc(
            logger,
            internal_model,
            initializer if warm_start else lambda m: m,
            (X_train, y_train),
            (X_test, y_test),
            build_metrics_tuple(metrics),
        )

        metrics = update_public_metrics(metrics, int_metrics)
        cur_w = internal_model.weights

        if unitize_data:
            internal_model.weights = normalized_into_input_space(
                internal_model.weights, col_norms
            )

        final_model = (
            update_public_model(model, internal_model)
            if return_convex
            else build_public_model(
                get_nc_formulation(internal_model, remove_sparse=True),
                model.bias,
            )
        )

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fname = os.path.join(save_path, str(reg))
            with open(fname, "wb") as f:
                pkl.dump((final_model, metrics), f)
            model_list.append(fname)
        else:
            model_list.append(deepcopy(final_model))

        metrics_list.append(deepcopy(metrics))
        internal_model.weights = cur_w  # restore for warm-start

    return model_list, metrics_list