# datasets

Deterministic synthetic demo data (NumPy, fixed seeds).

- `uv run python datasets/generate.py` -> `datasets/generated/clf_100k_16.csv` (3 classes, label column `label`) and
  `reg_100k_16.csv` (target column `target`). Git-ignored; each is under the 25 MB upload cap.
- `uv run python datasets/generate.py --fixtures` -> tiny committed fixtures in `fixtures/`.
- `--real` -> `fixtures/wine.csv` (classification, `label`) and `fixtures/diabetes.csv` (regression, `target`): real public datasets bundled with scikit-learn.

## Image datasets (CNN training)

`uv run python datasets/make_image_zip.py` builds `datasets/generated/fashion_mnist_10k.zip` (10 class folders of 28x28 PNGs, ~6 MB;
`--per-class 300` gives a 3k-image zip) from Fashion-MNIST, the dataset hosted on Kaggle as `zalando-research/fashionmnist`. Downloading
from Kaggle needs a login, so the script reads the identical official release; `--kaggle-csv fashion-mnist_train.csv` converts a Kaggle
download instead. Upload the zip in the app under **Image training**. Any zip of class folders (png/jpg) works, and a Kaggle MNIST-style
pixel CSV (label + pixel columns, <= 25 MB) is accepted too.
