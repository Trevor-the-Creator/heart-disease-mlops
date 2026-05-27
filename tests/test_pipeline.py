import pytest
import pandas as pd
import numpy as np
import os
from src.preprocess import load_and_clean_data, split_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# SETUP & FIXTURES
# ==========================================
# Point to the actual raw data for physiological validation
RAW_DATA_PATH = "data/raw/heart.csv"

@pytest.fixture
def sample_data_path(tmp_path):
    """Creates a tiny, temporary CSV to test logic without touching real data."""
    df = pd.DataFrame({
        'age': [63, 67, 67, 37, 41], 'sex': [1, 1, 1, 1, 0], 'cp': [1, 4, 4, 3, 2],
        'trestbps': [145, 160, 120, 130, 130], 'chol': [233, 286, 229, 250, 204],
        'fbs': [1, 0, 0, 0, 0], 'restecg': [2, 2, 2, 0, 2], 'thalach': [150, 108, 129, 187, 172],
        'exang': [0, 1, 1, 0, 0], 'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4], 'slope': [3, 2, 2, 3, 1],
        'ca': ['0.0', '3.0', '2.0', '?', '0.0'], 
        'thal': ['6.0', '3.0', '7.0', '3.0', '?'], 
        'target': [0, 2, 1, 0, 4] 
    })
    file_path = tmp_path / "dummy_heart.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def raw_data():
    """Loads the raw dataset strictly to validate its inherent properties."""
    df = pd.read_csv(RAW_DATA_PATH, na_values=["?"])
    # Apply standard column names if the raw file is missing headers
    if 'age' not in df.columns:
        df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    return df


# ==========================================
# 1. DATA VALIDATION TESTS 
# ==========================================
def test_expected_columns_are_present(raw_data):
    expected_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'target']
    for col in expected_cols:
        assert col in raw_data.columns, f"Critical column '{col}' is missing."

def test_target_values_are_valid(raw_data):
    valid_values = {0, 1, 2, 3, 4}
    actual_values = set(raw_data['target'].dropna().unique())
    assert actual_values.issubset(valid_values), "Target contains invalid bounds."

def test_numeric_ranges_are_physiological(raw_data):
    assert raw_data['age'].between(0, 120).all(), "Age values outside 0-120."
    assert (raw_data['chol'] > 0).all(), "Cholesterol must be > 0."


# ==========================================
# 2. PREPROCESSING LOGIC TESTS
# ==========================================
def test_handles_missing_values(sample_data_path):
    df = load_and_clean_data(sample_data_path)
    assert df['ca'].isna().sum() == 0  

def test_binarizes_target_variable(sample_data_path):
    df = load_and_clean_data(sample_data_path)
    assert set(df['target'].unique()).issubset({0, 1}) 

def test_split_data_shapes(sample_data_path):
    df = load_and_clean_data(sample_data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.4, random_state=42)
    # 5 dummy rows - 2 corrupted dropped rows = 3 clean rows (1 train, 2 test)
    assert len(X_train) == 1
    assert len(X_test) == 2

def test_does_not_modify_original_dataframe(sample_data_path):
    df = load_and_clean_data(sample_data_path)
    original_shape = df.shape
    split_data(df, test_size=0.2, random_state=42)
    assert df.shape == original_shape 

def test_invalid_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_and_clean_data("fake_directory/non_existent_file.csv")

def test_invalid_test_size_raises_error(sample_data_path):
    df = load_and_clean_data(sample_data_path)
    with pytest.raises(ValueError):
        split_data(df, test_size=1.5, random_state=42) 


# ==========================================
# 3. CATEGORICAL ENCODING TEST
# ==========================================
def test_categorical_encoding(tmp_path):
    mock_data = pd.DataFrame({
        "age": [50, 60, 45],
        "cp": [1, 2, 3],  
        "target": [1, 0, 1]
    })
    mock_file = tmp_path / "mock_heart.csv"
    mock_data.to_csv(mock_file, index=False)
    processed_df = load_and_clean_data(str(mock_file))
    
    assert "cp" not in processed_df.columns or any("cp_" in col for col in processed_df.columns)


# ==========================================
# 4. MODEL VALIDATION TESTS
# ==========================================
def test_model_prediction_type_and_shape(sample_data_path):
    df = load_and_clean_data(sample_data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.4, random_state=42)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(y_test) 
    assert set(preds).issubset({0, 1}) 

def test_model_minimum_accuracy():
    # Uses the real, preprocessed dataset to ensure the model actually learns
    df = load_and_clean_data(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    assert accuracy > 0.70