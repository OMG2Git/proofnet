import numpy as np
import pandas as pd
import pytest
from helpers import clf_frame, reg_frame

from proofnet_kernels.server import gaussian_nb as gnb
from proofnet_kernels.server import linear_ridge as ridge
from proofnet_kernels.server.common import KernelValidationError, profile_dataframe
from proofnet_kernels.server.params import GaussianNBParams, LinearRidgeParams
from proofnet_kernels.server.registry import REGISTRY, get_kernel


def _g(df: pd.DataFrame, **kw: object) -> GaussianNBParams:
    feats = [c for c in df.columns if c.startswith("f")]
    return GaussianNBParams(target_column="label", feature_columns=feats, **kw)


def _errors(df: pd.DataFrame, params: GaussianNBParams) -> list[str]:
    return gnb.validate(profile_dataframe(df), params).errors


def test_ok() -> None:
    df = clf_frame(500, 3, 3, 0)
    assert gnb.validate(profile_dataframe(df), _g(df)).ok


def test_row_limits() -> None:
    assert any("at least" in e for e in _errors(clf_frame(50, 2, 2, 0), _g(clf_frame(50, 2, 2, 0))))
    prof = profile_dataframe(clf_frame(200, 2, 2, 0))
    prof["n_rows"] = 300_001
    df = clf_frame(200, 2, 2, 0)
    assert any("at most" in e for e in gnb.validate(prof, _g(df)).errors)


def test_missing_columns_and_target_in_features() -> None:
    df = clf_frame(200, 2, 2, 0)
    p = GaussianNBParams(target_column="nope", feature_columns=["f0"])
    assert any("not found" in e for e in _errors(df, p))
    p = GaussianNBParams(target_column="label", feature_columns=["f0", "label"])
    assert any("must not be a feature" in e for e in _errors(df, p))


def test_non_numeric_feature_rejected() -> None:
    df = clf_frame(200, 2, 2, 0)
    df["f1"] = "a"
    assert any("not numeric" in e for e in _errors(df, _g(df)))


def test_missing_values_policy() -> None:
    df = clf_frame(300, 2, 2, 0)
    df.loc[0, "f0"] = np.nan
    assert any("reject" in e for e in _errors(df, _g(df, missing_values="reject")))
    prof = gnb.validate(profile_dataframe(df), _g(df))
    assert prof.ok and prof.warnings
    assert gnb.prepare(df, _g(df)).n_dropped == 1


def test_class_count_bounds() -> None:
    df = clf_frame(500, 2, 2, 0)
    df["label"] = 0
    assert any("classes" in e for e in _errors(df, _g(df)))
    df = clf_frame(5000, 2, 60, 0)
    assert any("classes" in e for e in _errors(df, _g(df)))


def test_class_with_too_few_rows() -> None:
    df = clf_frame(300, 2, 2, 0)
    df.loc[df.index[:], "label"] = 0
    df.loc[0, "label"] = 1  # a single row of class 1
    with pytest.raises(KernelValidationError):
        gnb.prepare(df, _g(df))


def test_all_constant_rejected_some_constant_warns() -> None:
    df = clf_frame(300, 2, 2, 0)
    df["f0"] = 1.0
    rep = gnb.validate(profile_dataframe(df), _g(df))
    assert rep.ok and rep.warnings
    df["f1"] = 1.0
    assert any("constant" in e for e in _errors(df, _g(df)))


def test_regression_target_must_be_numeric() -> None:
    df = reg_frame(300, 2, 0)
    df["target"] = "x"
    p = LinearRidgeParams(target_column="target", feature_columns=["f0", "f1"])
    assert any("numeric" in e for e in ridge.validate(profile_dataframe(df), p).errors)


def test_param_bounds() -> None:
    with pytest.raises(ValueError):
        LinearRidgeParams(target_column="t", feature_columns=["a"], alpha=-1)
    with pytest.raises(ValueError):
        GaussianNBParams(target_column="t", feature_columns=["a"], test_fraction=0.9)
    with pytest.raises(ValueError):
        GaussianNBParams(target_column="t", feature_columns=[f"f{i}" for i in range(65)])


def test_prepare_is_deterministic_and_pickle_free() -> None:
    df = clf_frame(500, 3, 3, 1)
    a, b = gnb.prepare(df, _g(df)), gnb.prepare(df, _g(df))
    assert a.sha256() == b.sha256()
    assert a.n_train + len(a.y_test) == 500
    assert a.class_labels == [0, 1, 2]
    from proofnet_kernels.server.common import load_prepared

    back = load_prepared(a.to_npz_bytes(), a.feature_names, a.class_labels)
    np.testing.assert_array_equal(back.x_train, a.x_train)


def test_registry() -> None:
    assert set(REGISTRY) == {"gaussian_nb_train@1", "linear_ridge_train@1", "cnn_image_train@1"}
    with pytest.raises(KeyError):
        get_kernel("evil@1")


def test_prepared_npz_is_byte_reproducible_across_time() -> None:
    import time

    df = clf_frame(400, 3, 3, 7)
    a = gnb.prepare(df, _g(df)).to_npz_bytes()
    time.sleep(2.2)  # zip timestamps have 2 s resolution; np.savez would differ here
    b = gnb.prepare(df, _g(df)).to_npz_bytes()
    assert a == b
