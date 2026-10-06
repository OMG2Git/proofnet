"""Shared backend-side kernel machinery: profiling, validation, preparation, compare."""

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.serialize import sha256_hex
from .params import CommonParams, MissingValuePolicy

MIN_ROWS, MAX_ROWS = 100, 300_000
MAX_FEATURES = 64
MIN_CLASSES, MAX_CLASSES = 2, 50


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class KernelValidationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        super().__init__("; ".join(report.errors))
        self.report = report


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    cols: list[dict[str, Any]] = []
    for name in df.columns:
        s = df[name]
        numeric = bool(pd.api.types.is_numeric_dtype(s)) and not pd.api.types.is_bool_dtype(s)
        cols.append(
            {
                "name": str(name),
                "dtype": "numeric" if numeric else "text",
                "missing": int(s.isna().sum()),
                "unique": int(s.nunique(dropna=True)),
                "min": float(s.min()) if numeric and s.notna().any() else None,
                "max": float(s.max()) if numeric and s.notna().any() else None,
            }
        )
    return {"n_rows": int(len(df)), "columns": cols}


def validate_profile(
    profile: dict[str, Any], params: CommonParams, *, regression: bool
) -> ValidationReport:
    """Dataset-profile checks from ARCHITECTURE 5.1 (data-level checks run again in prepare)."""
    errors: list[str] = []
    warnings: list[str] = []
    by_name = {c["name"]: c for c in profile["columns"]}
    n_rows = profile["n_rows"]
    if n_rows < MIN_ROWS:
        errors.append(f"dataset has {n_rows} rows; at least {MIN_ROWS} required")
    if n_rows > MAX_ROWS:
        errors.append(f"dataset has {n_rows} rows; at most {MAX_ROWS} allowed")
    feats = params.feature_columns
    if len(set(feats)) != len(feats):
        errors.append("feature_columns contains duplicates")
    if not 1 <= len(feats) <= MAX_FEATURES:
        errors.append(f"between 1 and {MAX_FEATURES} feature columns required")
    if params.target_column in feats:
        errors.append("target_column must not be a feature column")
    target = by_name.get(params.target_column)
    if target is None:
        errors.append(f"target column '{params.target_column}' not found")
    used = [c for c in [params.target_column, *feats] if c in by_name]
    for f in feats:
        col = by_name.get(f)
        if col is None:
            errors.append(f"feature column '{f}' not found")
        elif col["dtype"] != "numeric":
            errors.append(f"feature column '{f}' is not numeric (categorical encoding unsupported)")
    n_missing = sum(by_name[c]["missing"] for c in used)
    if n_missing:
        if params.missing_values is MissingValuePolicy.reject:
            errors.append(f"{n_missing} missing values and missing_values='reject'")
        else:
            warnings.append(f"{n_missing} missing cells; affected rows will be dropped")
    if target is not None:
        if regression and target["dtype"] != "numeric":
            errors.append("regression target must be numeric")
        if not regression and not MIN_CLASSES <= target["unique"] <= MAX_CLASSES:
            errors.append(
                f"classification target needs {MIN_CLASSES}-{MAX_CLASSES} classes; "
                f"found {target['unique']}"
            )
    numeric_feats = [by_name[f] for f in feats if f in by_name and by_name[f]["dtype"] == "numeric"]
    const = [c["name"] for c in numeric_feats if c["min"] is not None and c["min"] == c["max"]]
    if numeric_feats and len(const) == len(numeric_feats):
        errors.append("all feature columns are constant")
    elif const:
        warnings.append(f"constant feature columns: {const}")
    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        summary={"n_rows": n_rows, "n_features": len(feats)},
    )


@dataclass
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_labels: list[Any]  # empty for regression
    n_dropped: int = 0

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.x_train.shape[1])

    def to_npz_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez(
            buf,
            X_train=self.x_train,
            y_train=self.y_train,
            X_test=self.x_test,
            y_test=self.y_test,
        )
        return buf.getvalue()

    def sha256(self) -> str:
        return sha256_hex(self.to_npz_bytes())


def prepare_dataframe(df: pd.DataFrame, params: CommonParams, *, regression: bool) -> PreparedData:
    report = validate_profile(profile_dataframe(df), params, regression=regression)
    if not report.ok:
        raise KernelValidationError(report)
    cols = [params.target_column, *params.feature_columns]
    sub = df[cols]
    n_before = len(sub)
    sub = sub.dropna()
    n_dropped = n_before - len(sub)
    x = sub[params.feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise KernelValidationError(ValidationReport(False, ["non-finite feature values"]))
    labels: list[Any] = []
    if regression:
        y = sub[params.target_column].to_numpy(dtype=np.float64)
        if not np.isfinite(y).all():
            raise KernelValidationError(ValidationReport(False, ["non-finite target values"]))
    else:
        uniques, y = np.unique(sub[params.target_column].to_numpy(), return_inverse=True)
        y = y.astype(np.int64)
        labels = [_label(u) for u in uniques]
        if not MIN_CLASSES <= len(labels) <= MAX_CLASSES:
            raise KernelValidationError(
                ValidationReport(False, [f"{len(labels)} classes after dropping missing rows"])
            )
    if len(x) < MIN_ROWS:
        raise KernelValidationError(
            ValidationReport(False, [f"only {len(x)} usable rows (need {MIN_ROWS})"])
        )
    perm = np.random.default_rng(params.split_seed).permutation(len(x))
    n_test = max(1, round(len(x) * params.test_fraction))
    te, tr = perm[:n_test], perm[n_test:]
    if not regression:
        counts = np.bincount(y[tr], minlength=len(labels))
        if (counts < 2).any():
            raise KernelValidationError(
                ValidationReport(False, ["every class needs at least 2 training rows"])
            )
    return PreparedData(
        x_train=x[tr],
        y_train=y[tr],
        x_test=x[te],
        y_test=y[te],
        feature_names=list(params.feature_columns),
        class_labels=labels,
        n_dropped=n_dropped,
    )


def load_prepared(data: bytes, feature_names: list[str], class_labels: list[Any]) -> PreparedData:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return PreparedData(
            z["X_train"], z["y_train"], z["X_test"], z["y_test"], feature_names, class_labels
        )


def _label(v: Any) -> Any:
    if isinstance(v, (np.floating, float)) and float(v).is_integer():
        return int(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def compare_states(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Normwise relative discrepancy per field: max|a-b| / max|b| (abs diff if b == 0).

    Reused by Part 2 verification. Returns {"max_rel_diff": float, "fields": {path: float}}.
    """
    fields: dict[str, float] = {}

    def walk(path: str, x: Any, y: Any) -> None:
        if isinstance(x, dict):
            for k in x:
                walk(f"{path}.{k}" if path else k, x[k], y[k])
            return
        xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if xa.shape != ya.shape:
            fields[path] = float("inf")
            return
        scale = float(np.max(np.abs(ya))) if ya.size else 0.0
        diff = float(np.max(np.abs(xa - ya))) if xa.size else 0.0
        fields[path] = diff / scale if scale > 0 else diff

    walk("", a, b)
    return {"max_rel_diff": max(fields.values(), default=0.0), "fields": fields}
