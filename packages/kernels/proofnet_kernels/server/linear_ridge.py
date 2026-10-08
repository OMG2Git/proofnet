"""Linear/Ridge regression kernel, backend side. Worker side lives in core/linear_ridge.py."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..core.serialize import all_finite
from .common import (
    PreparedData,
    ValidationReport,
    compare_states,
    prepare_dataframe,
    validate_profile,
)
from .params import LinearRidgeParams

PARAMS_MODEL = LinearRidgeParams
AUDIT_FLOOR = 1e-8  # smallest tolerance used by the trust system
TOLERANCE = 1e-6  # normwise relative, distributed vs centralized


@dataclass
class FinalResult:
    model: Ridge
    model_json: dict[str, Any]
    metrics: dict[str, Any]
    y_pred: np.ndarray


def validate(profile: dict[str, Any], params: LinearRidgeParams) -> ValidationReport:
    return validate_profile(profile, params, regression=True)


def prepare(df: pd.DataFrame, params: LinearRidgeParams) -> PreparedData:
    return prepare_dataframe(df, params, regression=True)


def worker_params(prepared: PreparedData) -> dict[str, Any]:
    return {}


def validate_partial(partial: dict[str, Any], n_rows: int, n_features: int) -> list[str]:
    try:
        d = n_features
        if np.asarray(partial["mean_x"]).shape != (d,):
            return ["mean_x has wrong shape"]
        if np.asarray(partial["Sxx"]).shape != (d, d):
            return ["Sxx has wrong shape"]
        if np.asarray(partial["Sxy"]).shape != (d,):
            return ["Sxy has wrong shape"]
        if not all_finite(
            [
                partial["mean_x"],
                partial["mean_y"],
                partial["Sxx"],
                partial["Sxy"],
                partial["Syy"],
            ]
        ):
            return ["non-finite or non-numeric values"]
        if not isinstance(partial["n"], int) or partial["n"] != n_rows:
            return [f"n does not match chunk rows ({n_rows})"]
        if partial["Syy"] < 0 or np.diag(np.asarray(partial["Sxx"])).min() < 0:
            return ["negative second moments"]
    except (KeyError, TypeError, ValueError) as e:
        return [f"malformed payload: {e}"]
    return []


def solve(merged: dict[str, Any], alpha: float) -> tuple[np.ndarray, float]:
    sxx = np.asarray(merged["Sxx"], dtype=np.float64)
    sxy = np.asarray(merged["Sxy"], dtype=np.float64)
    if alpha == 0:
        beta = np.linalg.lstsq(sxx, sxy, rcond=None)[0]
    else:
        try:
            beta = np.linalg.solve(sxx + alpha * np.eye(sxx.shape[0]), sxy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(sxx + alpha * np.eye(sxx.shape[0]), sxy, rcond=None)[0]
    intercept = float(merged["mean_y"] - np.asarray(merged["mean_x"]) @ beta)
    return beta, intercept


def finalize(
    merged: dict[str, Any], prepared: PreparedData, params: LinearRidgeParams
) -> FinalResult:
    beta, intercept = solve(merged, params.alpha)
    model = Ridge(alpha=params.alpha)
    model.coef_ = beta
    model.intercept_ = intercept
    model.n_features_in_ = beta.shape[0]
    y_pred = model.predict(prepared.x_test)
    metrics = {
        "r2": float(r2_score(prepared.y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(prepared.y_test, y_pred))),
        "mae": float(mean_absolute_error(prepared.y_test, y_pred)),
        "n_test": int(len(prepared.y_test)),
    }
    model_json = {
        "kind": "linear_ridge",
        "feature_names": prepared.feature_names,
        "alpha": params.alpha,
        "coef": beta.tolist(),
        "intercept": intercept,
    }
    return FinalResult(model, model_json, metrics, y_pred)


def reference(prepared: PreparedData) -> dict[str, Any]:
    """Centralized merged state via an independent formulation (np.cov / np.var)."""
    x, y = prepared.x_train, prepared.y_train
    n = x.shape[0]
    sxx = np.atleast_2d(np.cov(x.T, bias=True)) * n
    yc = y - y.mean()
    xc = x - x.mean(axis=0)
    return {
        "n": int(n),
        "mean_x": x.mean(axis=0).tolist(),
        "mean_y": float(y.mean()),
        "Sxx": sxx.tolist(),
        "Sxy": (xc * yc[:, None]).sum(axis=0).tolist(),
        "Syy": float(y.var() * n),
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = compare_states(a, b)
    out["tolerance"] = TOLERANCE
    out["within_tolerance"] = out["max_rel_diff"] <= TOLERANCE
    return out
