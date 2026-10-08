"""Gaussian Naive Bayes kernel, backend side: validate/prepare/validate_partial/finalize/
reference/compare. Worker side (map, merge) lives in core/gaussian_nb.py."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB

from ..core.serialize import all_finite
from .common import (
    PreparedData,
    ValidationReport,
    compare_states,
    prepare_dataframe,
    validate_profile,
)
from .params import GaussianNBParams

PARAMS_MODEL = GaussianNBParams
AUDIT_FLOOR = 1e-10  # smallest tolerance used by the trust system
TOLERANCE = 1e-8  # normwise relative, distributed vs centralized


@dataclass
class FinalResult:
    model: GaussianNB
    model_json: dict[str, Any]
    metrics: dict[str, Any]
    y_pred: np.ndarray


def validate(profile: dict[str, Any], params: GaussianNBParams) -> ValidationReport:
    return validate_profile(profile, params, regression=False)


def prepare(df: pd.DataFrame, params: GaussianNBParams) -> PreparedData:
    return prepare_dataframe(df, params, regression=False)


def worker_params(prepared: PreparedData) -> dict[str, Any]:
    return {"n_classes": len(prepared.class_labels)}


def validate_partial(
    partial: dict[str, Any], n_rows: int, n_features: int, n_classes: int
) -> list[str]:
    """Structural check; returns a list of errors (empty = ok)."""
    try:
        cls = partial["classes"]
        shapes = {
            "mean": (n_features,),
            "M2": (n_features,),
        }
        errs: list[str] = []
        for k, shape in shapes.items():
            if np.asarray(partial[k]).shape != shape:
                errs.append(f"{k} has wrong shape")
        for k in ("mean", "M2"):
            if np.asarray(cls[k]).shape != (n_classes, n_features):
                errs.append(f"classes.{k} has wrong shape")
        if np.asarray(cls["n"]).shape != (n_classes,):
            errs.append("classes.n has wrong shape")
        if errs:
            return errs
        if not all_finite([partial["mean"], partial["M2"], cls["mean"], cls["M2"], cls["n"]]):
            return ["non-finite or non-numeric values"]
        counts = cls["n"]
        if any(not isinstance(c, int) or isinstance(c, bool) or c < 0 for c in counts):
            return ["class counts must be non-negative integers"]
        if partial["n"] != n_rows or sum(counts) != n_rows:
            return [f"counts do not match chunk rows ({n_rows})"]
        if np.asarray(partial["M2"]).min() < 0 or np.asarray(cls["M2"]).min() < 0:
            return ["negative M2"]
    except (KeyError, TypeError, ValueError) as e:
        return [f"malformed payload: {e}"]
    return []


def finalize(
    merged: dict[str, Any], prepared: PreparedData, params: GaussianNBParams
) -> FinalResult:
    cls = merged["classes"]
    n_c = np.asarray(cls["n"], dtype=np.float64)
    theta = np.asarray(cls["mean"], dtype=np.float64)
    var = np.asarray(cls["M2"], dtype=np.float64) / n_c[:, None]
    # sklearn: epsilon = var_smoothing * max(np.var(X, axis=0)) over all training rows
    epsilon = params.var_smoothing * float(np.max(np.asarray(merged["M2"]) / merged["n"]))
    var = var + epsilon
    model = GaussianNB(var_smoothing=params.var_smoothing)
    model.classes_ = np.arange(len(n_c))
    model.class_count_ = n_c
    model.class_prior_ = n_c / n_c.sum()
    model.theta_ = theta
    model.var_ = var
    model.epsilon_ = epsilon
    model.n_features_in_ = theta.shape[1]
    y_pred = model.predict(prepared.x_test)
    labels = np.arange(len(n_c))
    metrics = {
        "accuracy": float(accuracy_score(prepared.y_test, y_pred)),
        "macro_f1": float(f1_score(prepared.y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(prepared.y_test, y_pred, labels=labels).tolist(),
        "n_test": int(len(prepared.y_test)),
    }
    model_json = {
        "kind": "gaussian_nb",
        "classes": prepared.class_labels,
        "feature_names": prepared.feature_names,
        "class_prior": model.class_prior_.tolist(),
        "theta": theta.tolist(),
        "var": var.tolist(),
        "epsilon": epsilon,
    }
    return FinalResult(model, model_json, metrics, y_pred)


def reference(prepared: PreparedData, n_classes: int) -> dict[str, Any]:
    """Centralized merged state via an independent NumPy formulation (np.mean/np.var)."""
    x, y = prepared.x_train, prepared.y_train
    n = x.shape[0]
    cn, cmean, cm2 = [], [], []
    for c in range(n_classes):
        xc = x[y == c]
        cn.append(int(xc.shape[0]))
        cmean.append(xc.mean(axis=0).tolist())
        cm2.append((xc.var(axis=0) * xc.shape[0]).tolist())
    return {
        "n": int(n),
        "mean": x.mean(axis=0).tolist(),
        "M2": (x.var(axis=0) * n).tolist(),
        "classes": {"n": cn, "mean": cmean, "M2": cm2},
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = compare_states(a, b)
    out["tolerance"] = TOLERANCE
    out["within_tolerance"] = out["max_rel_diff"] <= TOLERANCE
    return out
