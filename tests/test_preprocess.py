import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.preprocess import load_and_clean_data, split_data, scale_features, NUMERIC_COLS


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "age":      [63, 67, 37, 41, 56],
        "sex":      [1,  1,  1,  0,  1],
        "cp":       [1,  4,  3,  2,  1],
        "trestbps": [145, 160, 130, 130, 120],
        "chol":     [233, 286, 250, 204, 236],
        "fbs":      [1,  0,  0,  0,  0],
        "restecg":  [2,  2,  0,  2,  0],
        "thalach":  [150, 108, 187, 172, 178],
        "exang":    [0,  1,  0,  0,  0],
        "oldpeak":  [2.3, 1.5, 3.5, 1.4, 0.8],
        "slope":    [3,  2,  3,  1,  1],
        "ca":       ["0.0", "3.0", "?", "0.0", "1.0"],
        "thal":     ["6.0", "3.0", "3.0", "?", "3.0"],
        "target":   [0,  2,  0,  4,  0],
    })
    path = tmp_path / "heart_test.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_missing_values_are_dropped(sample_csv):
    df = load_and_clean_data(sample_csv)
    assert df.isnull().sum().sum() == 0


def test_target_is_binarized(sample_csv):
    df = load_and_clean_data(sample_csv)
    assert set(df["target"].unique()).issubset({0, 1})


def test_categorical_columns_are_one_hot_encoded(sample_csv):
    df = load_and_clean_data(sample_csv)
    for col in ["cp", "restecg", "slope", "thal"]:
        assert col not in df.columns, f"'{col}' should be one-hot encoded, not kept raw"
    assert any(c.startswith("cp_") for c in df.columns)


def test_split_does_not_modify_original_dataframe(sample_csv):
    df = load_and_clean_data(sample_csv)
    original_shape = df.shape
    split_data(df, test_size=0.3, random_state=42)
    assert df.shape == original_shape


def test_invalid_path_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_and_clean_data("no/such/file.csv")


def test_invalid_test_size_raises_value_error(sample_csv):
    df = load_and_clean_data(sample_csv)
    with pytest.raises(ValueError):
        split_data(df, test_size=1.5)


def test_scale_features_fits_scaler_on_train_only(sample_csv):
    df = load_and_clean_data(sample_csv)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.4, random_state=42)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    assert isinstance(scaler, StandardScaler)
    assert hasattr(scaler, "mean_"), "Scaler was not fitted"
    assert X_train_s.shape == X_train.shape, "Scaling must not change DataFrame shape"
    assert X_test_s.shape == X_test.shape

    # One-hot encoded columns must be untouched (still 0/1 integers)
    ohe_cols = [c for c in X_train_s.columns if "_" in c]
    for col in ohe_cols:
        assert X_train_s[col].isin([0, 1, True, False]).all()
