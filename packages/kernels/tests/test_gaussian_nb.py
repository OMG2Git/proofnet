import numpy as np
import pytest
from helpers import clf_frame, partition, roundtrip
from sklearn.naive_bayes import GaussianNB

from proofnet_kernels.core import gaussian_nb as core
from proofnet_kernels.server import gaussian_nb as srv
from proofnet_kernels.server.params import GaussianNBParams


def _params(d: int, **kw: object) -> GaussianNBParams:
    return GaussianNBParams(
        target_column="label",
        feature_columns=[f"f{i}" for i in range(d)],
        **kw,
    )


def _distributed(prepared: object, ranges: list[tuple[int, int]], c: int) -> dict:  # type: ignore[type-arg]
    x, y = prepared.x_train, prepared.y_train  # type: ignore[attr-defined]
    parts = [roundtrip(core.map(x[a:b], y[a:b], c)) for a, b in ranges]
    return core.merge(parts)


@pytest.mark.parametrize("k", [1, 2, 3, 7])
@pytest.mark.parametrize("seed", range(5))
def test_distributed_equals_centralized(k: int, seed: int) -> None:
    d, c = 6, 4
    params = _params(d, var_smoothing=1e-9)
    prep = srv.prepare(clf_frame(2000, d, c, seed), params)
    ranges = partition(prep.n_train, k, np.random.default_rng(seed))
    merged = _distributed(prep, ranges, c)
    # state equals independent centralized state
    cmp = srv.compare(merged, srv.reference(prep, c))
    assert cmp["within_tolerance"], cmp["max_rel_diff"]
    # finalized model equals sklearn centralized fit
    res = srv.finalize(merged, prep, params)
    ref = GaussianNB(var_smoothing=1e-9).fit(prep.x_train, prep.y_train)
    np.testing.assert_allclose(res.model.theta_, ref.theta_, rtol=1e-8)
    np.testing.assert_allclose(res.model.var_, ref.var_, rtol=1e-8)
    np.testing.assert_allclose(res.model.class_prior_, ref.class_prior_, rtol=1e-12)
    assert (res.model.predict(prep.x_test) == ref.predict(prep.x_test)).all()
    assert (res.y_pred == ref.predict(prep.x_test)).all()


def test_chunks_missing_classes() -> None:
    d, c = 3, 3
    params = _params(d)
    prep = srv.prepare(clf_frame(600, d, c, 1), params)
    order = np.argsort(prep.y_train, kind="stable")  # each chunk sees ~one class
    x, y = prep.x_train[order], prep.y_train[order]
    ranges = partition(len(y), 5, np.random.default_rng(3))
    merged = core.merge([core.map(x[a:b], y[a:b], c) for a, b in ranges])
    prep.x_train, prep.y_train = x, y
    assert srv.compare(merged, srv.reference(prep, c))["within_tolerance"]


def test_single_row_chunks_and_constant_feature() -> None:
    d, c = 3, 2
    df = clf_frame(300, d, c, 5)
    df["f1"] = 4.0  # constant feature (warning, not error)
    params = _params(d)
    prep = srv.prepare(df, params)
    ranges = [(0, 1), (1, 2), (2, prep.n_train)]
    merged = _distributed(prep, ranges, c)
    assert srv.compare(merged, srv.reference(prep, c))["within_tolerance"]
    ref = GaussianNB().fit(prep.x_train, prep.y_train)
    res = srv.finalize(merged, prep, params)
    np.testing.assert_allclose(res.model.var_, ref.var_, rtol=1e-8)


def test_two_and_fifty_classes() -> None:
    for c in (2, 50):
        d = 4
        params = _params(d)
        prep = srv.prepare(clf_frame(20000, d, c, 9), params)
        ranges = partition(prep.n_train, 3, np.random.default_rng(1))
        merged = _distributed(prep, ranges, c)
        assert srv.compare(merged, srv.reference(prep, c))["within_tolerance"]
        res = srv.finalize(merged, prep, params)
        ref = GaussianNB().fit(prep.x_train, prep.y_train)
        assert (res.model.predict(prep.x_test) == ref.predict(prep.x_test)).all()


def test_large_offset_numerical_stability() -> None:
    d, c = 2, 2
    df = clf_frame(1000, d, c, 2)
    df[["f0", "f1"]] += 1e8  # catastrophic for naive sum-of-squares
    params = _params(d)
    prep = srv.prepare(df, params)
    merged = _distributed(prep, partition(prep.n_train, 4, np.random.default_rng(0)), c)
    assert srv.compare(merged, srv.reference(prep, c))["within_tolerance"]


def test_validate_partial() -> None:
    d, c = 3, 2
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(50, d)), rng.integers(0, c, size=50)
    good = roundtrip(core.map(x, y, c))
    assert srv.validate_partial(good, 50, d, c) == []
    assert srv.validate_partial(good, 51, d, c)  # wrong row count
    assert srv.validate_partial(good, 50, d + 1, c)  # wrong shape
    bad = roundtrip(core.map(x, y, c))
    bad["mean"][0] = float("inf")
    assert srv.validate_partial(bad, 50, d, c)
    bad = roundtrip(core.map(x, y, c))
    bad["classes"]["M2"][0][0] = -1.0
    assert srv.validate_partial(bad, 50, d, c)
    bad = roundtrip(core.map(x, y, c))
    del bad["classes"]
    assert srv.validate_partial(bad, 50, d, c)


def test_compare_detects_corruption() -> None:
    d, c = 3, 2
    prep = srv.prepare(clf_frame(500, d, c, 4), _params(d))
    merged = _distributed(prep, [(0, prep.n_train)], c)
    merged["classes"]["mean"][0][0] *= 1.001
    assert not srv.compare(merged, srv.reference(prep, c))["within_tolerance"]


def test_joblib_roundtrip_predicts_identically(tmp_path: object) -> None:
    import joblib

    d, c = 5, 3
    params = _params(d)
    prep = srv.prepare(clf_frame(3000, d, c, 11), params)
    merged = _distributed(prep, partition(prep.n_train, 3, np.random.default_rng(2)), c)
    res = srv.finalize(merged, prep, params)
    path = tmp_path / "m.joblib"  # type: ignore[operator]
    joblib.dump(res.model, path)
    loaded = joblib.load(path)
    ref = GaussianNB().fit(prep.x_train, prep.y_train)
    assert (loaded.predict(prep.x_test) == ref.predict(prep.x_test)).all()
    np.testing.assert_allclose(
        loaded.predict_proba(prep.x_test), ref.predict_proba(prep.x_test), rtol=1e-7, atol=1e-12
    )
