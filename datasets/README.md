# datasets

Deterministic synthetic demo data (NumPy, fixed seeds).

- `uv run python datasets/generate.py` -> `datasets/generated/clf_100k_16.csv` (3 classes, label column `label`) and
  `reg_100k_16.csv` (target column `target`). Git-ignored; each is under the 25 MB upload cap.
- `uv run python datasets/generate.py --fixtures` -> tiny committed fixtures in `fixtures/`.
- `--real` -> `fixtures/wine.csv` (classification, `label`) and `fixtures/diabetes.csv` (regression, `target`): real public datasets bundled with scikit-learn.
