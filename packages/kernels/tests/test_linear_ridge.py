import numpy as np
import pytest
from helpers import partition, reg_frame, roundtrip
from sklearn.linear_model import LinearRegression, Ridge

from proofnet_kernels.core import linear_ridge as core
from proofnet_kernels.server import linear_ridge as srv
from proofnet_kernels.server.params import LinearRidgeParams


def _params(d: int, alpha: float = 1.0) -> LinearRidgeParams:
    return LinearRidgeParams(
        target_column="target", feature_columns=[f"f{i}" for i in range(d)], alpha=alpha
    )


def _merged(prep: object, ranges: list[tuple[int, int]]) -> dict:  # type: ignore[type-arg]
    x, y = prep.x_train, prep.y_train  # type: ignore[attr-defined]
    return core.merge([roundtrip(core.map(x[a:b], y[a:b])) for a, b in ranges])


@pytest.mark.parametrize("alpha", [0.0, 0.5, 10.0])
@pytest.mark.parametrize("k", [1, 2, 3, 7])
@pytest.mark.parametrize("seed", range(4))
def test_distributed_equals_centralized(alpha: float, k: int, seed: int) -> None:
    d = 8
    params = _params(d, alpha)
    prep = srv.prepare(reg_frame(3000, d, seed), params)
    merged = _merged(prep, partition(prep.n_train, k, np.random.default_rng(seed)))
    assert srv.compare(merged, srv.reference(prep))["within_tolerance"]
    res = srv.finalize(merged, prep, params)
    ref = (Ridge(alpha=alpha) if alpha else LinearRegression()).fit(prep.x_train, prep.y_train)
    np.testing.assert_allclose(res.model.coef_, ref.coef_, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(res.model.intercept_, ref.intercept_, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(res.y_pred, ref.predict(prep.x_test), rtol=1e-6, atol=1e-8)


def test_single_row_chunks() -> None:
    d = 3
    params = _params(d)
    prep = srv.prepare(reg_frame(300, d, 3), params)
    merged = _merged(prep, [(0, 1), (1, 2), (2, prep.n_train)])
    assert srv.compare(merged, srv.reference(prep))["within_tolerance"]


def test_alpha_zero_singular_falls_back_to_lstsq() -> None:
    d = 3
    df = reg_frame(400, d, 6)
    df["f2"] = df["f0"] * 2.0  # exactly collinear
    params = _params(d, 0.0)
    prep = srv.prepare(df, params)
    merged = _merged(prep, partition(prep.n_train, 3, np.random.default_rng(0)))
    res = srv.finalize(merged, prep, params)
    ref = LinearRegression().fit(prep.x_train, prep.y_train)
    np.testing.assert_allclose(res.y_pred, ref.predict(prep.x_test), rtol=1e-6, atol=1e-6)


def test_single_feature() -> None:
    params = _params(1)
    prep = srv.prepare(reg_frame(500, 1, 8), params)
    merged = _merged(prep, partition(prep.n_train, 3, np.random.default_rng(0)))
    assert srv.compare(merged, srv.reference(prep))["within_tolerance"]


def test_validate_partial() -> None:
    d = 3
    rng = np.random.default_rng(0)
    good = roundtrip(core.map(rng.normal(size=(40, d)), rng.normal(size=40)))
    assert srv.validate_partial(good, 40, d) == []
    assert srv.validate_partial(good, 41, d)
    assert srv.validate_partial(good, 40, d + 1)
    bad = roundtrip(good)
    bad["Syy"] = -1.0
    assert srv.validate_partial(bad, 40, d)
    bad = roundtrip(good)
    bad["Sxx"][0][0] = float("nan")
    assert srv.validate_partial(bad, 40, d)


def test_compare_detects_corruption() -> None:
    prep = srv.prepare(reg_frame(500, 3, 4), _params(3))
    merged = _merged(prep, [(0, prep.n_train)])
    merged["Sxy"][0] *= 1.01
    assert not srv.compare(merged, srv.reference(prep))["within_tolerance"]
