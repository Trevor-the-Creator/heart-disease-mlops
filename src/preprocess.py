"""
preprocess.py — Data cleaning, encoding, scaling, and splitting for the
UCI Cleveland Heart Disease dataset.

Pipeline summary
----------------
1. Load CSV, treating '?' as NaN (missing sentinel used in the original data).
2. Auto-assign standard column names if the file has no header row.
3. Drop rows with any remaining missing values (6 rows in the raw Cleveland file).
4. Binarize the target: original values 1–4 all indicate disease present → 1;
   value 0 indicates no disease → 0.
5. One-hot encode four nominal categorical features (cp, restecg, slope, thal)
   with drop_first=True to avoid the dummy-variable trap.
6. After train/test split, fit StandardScaler on the continuous numeric columns
   of the TRAINING set only, then transform both sets — preventing data leakage
   from the test set into the learned scaling parameters.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Continuous numeric columns scaled to zero mean / unit variance.
# Binary indicators (sex, fbs, exang) and one-hot columns are intentionally
# excluded — scaling them would not improve tree-based models and could distort
# the interpretability of the logistic regression coefficients.
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]


def load_and_clean_data(file_path):
    """Load the heart disease CSV and apply all cleaning / encoding steps.

    Parameters
    ----------
    file_path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with binary target and one-hot encoded categoricals.
        Numeric columns are NOT yet scaled — call scale_features() after splitting.
    """
    # Step 1 — Load: '?' is the missing-value sentinel in the Cleveland dataset.
    df = pd.read_csv(file_path, na_values=["?"])

    # Step 2 — Header guard: some versions of the file ship without column names.
    if "age" not in df.columns:
        columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
        ]
        df = pd.read_csv(file_path, na_values=["?"], names=columns)

    # Step 3 — Drop missing rows.
    # The Cleveland file has 6 rows with '?' in 'ca' or 'thal'; dropping them
    # leaves 297 complete cases with no imputation bias introduced.
    df = df.dropna()

    # Step 4 — Binarize target: the raw target is 0–4 (severity level).
    # For binary classification we treat 0 as no disease and 1–4 as disease.
    if "target" in df.columns:
        df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    # Step 5 — One-hot encode nominal categoricals.
    # cp (chest pain type), restecg (resting ECG), slope (ST slope), and thal
    # (thalassemia) are nominal, not ordinal — encoding them as integers would
    # impose a false ordering on tree splits and regression coefficients.
    # drop_first=True removes one dummy per variable to avoid multicollinearity.
    categorical_cols = ["cp", "restecg", "slope", "thal"]
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    return df


def scale_features(X_train, X_test):
    """Fit StandardScaler on training data only, then transform both splits.

    Scaling is applied only to NUMERIC_COLS (continuous features).  Binary
    indicators and one-hot columns are left unchanged.  Fitting on X_train
    only prevents any test-set statistics from leaking into the scaler.

    Returns
    -------
    X_train_scaled, X_test_scaled : pd.DataFrame
    scaler : fitted StandardScaler (saved as an MLflow artifact for inference)
    """
    cols = [c for c in NUMERIC_COLS if c in X_train.columns]
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])
    return X_train, X_test, scaler


def split_data(df, test_size=0.2, random_state=42):
    """Split a preprocessed DataFrame into train and test feature/label sets.

    Parameters
    ----------
    df : pd.DataFrame   Cleaned, encoded DataFrame (output of load_and_clean_data).
    test_size : float   Fraction of data reserved for testing (0 < test_size < 1).
    random_state : int  Random seed for reproducibility.
    """
    if "target" in df.columns:
        X = df.drop(columns=["target"])
        y = df["target"]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
