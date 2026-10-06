# Kernels — math and tolerances

Both kernels use the sufficient-statistics pattern: workers compute mergeable moments of their row range; the backend merges them in chunk-index order and finalizes. Merges use Chan et al. parallel updates (no `sum(x^2) - sum(x)^2` cancellation). A state with `n = 0` is the identity, so chunks missing a class are fine.

## `gaussian_nb_train@1`

- **map** (`core/gaussian_nb.py`): overall `n, mean, M2` per feature and, per class, `n_c, mean_c, M2_c`.
- **merge:** `n = na+nb`, `delta = mb-ma`, `mean = ma + delta*nb/n`, `M2 = M2a + M2b + delta^2*na*nb/n`.
- **finalize** (`server/gaussian_nb.py`): `theta = mean_c`, `var = M2_c/n_c + eps`, `eps = var_smoothing * max(M2/n)`, `prior = n_c/n`, loaded into a scikit-learn `GaussianNB` (sklearn pinned to 1.9.1).
- **reference:** independent centralized state via `np.mean` / `np.var` (not the merge code). The test suite also compares the finalized model with `GaussianNB().fit`.
- **Tolerance:** normwise relative discrepancy `<= 1e-8` (`compare()`); observed ~1e-15 or exactly 0.

## `linear_ridge_train@1`

- **map** (`core/linear_ridge.py`): `n, mean_x, mean_y, Sxx, Sxy, Syy` (centered).
- **merge:** `Sxx = Sxx_a + Sxx_b + (na nb/n) dx dx^T`, `Sxy = ... + (na nb/n) dx dy`, `Syy = ... + (na nb/n) dy^2`.
- **finalize** (`server/linear_ridge.py`): solve `(Sxx + alpha I) beta = Sxy`; `intercept = mean_y - mean_x . beta`. `alpha = 0` uses least squares (minimum-norm, handles singular `Sxx`); a singular solve with `alpha > 0` also falls back to least squares.
- **reference:** independent state via `np.cov` / `np.var`; tests also compare with sklearn `Ridge` / `LinearRegression`.
- **Tolerance:** state `<= 1e-6` normwise relative; observed ~1e-15.

## Verification in this repo

- Equivalence for random partitions (1, 2, 3, 7 chunks, uneven sizes), chunks missing classes, single-row chunks, constant feature, 2 and 50 classes, `alpha = 0` collinear case, 1e8 offset stability, and corruption detection by `compare()`.
- Pyodide parity: `tests/test_pyodide_parity.py` runs `core` map inside Pyodide 314.0.7 (Python 3.14.2, NumPy 2.4.6) under Node and compares with CPython; the observed discrepancy is written to `pyodide-parity/last_discrepancy.json` and recorded in `CLAUDE.md` section 12. Needs `npm ci` in `packages/kernels/pyodide-parity`.
- `tests/test_core_imports.py` fails if anything in `core/` imports something other than NumPy or the standard library.
