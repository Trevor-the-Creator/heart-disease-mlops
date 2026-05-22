import pytest
import pandas as pd
from src.preprocess import load_and_clean_data, split_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# SETUP: Create dummy data for safe testing
# ==========================================
@pytest.fixture
def sample_data_path(tmp_path):
    """Creates a tiny, temporary CSV to test our functions without touching real data."""
    df = pd.DataFrame({
        'age': [63, 67, 67, 37, 41], 'sex': [1, 1, 1, 1, 0], 'cp': [1, 4, 4, 3, 2],
        'trestbps': [145, 160, 120, 130, 130], 'chol': [233, 286, 229, 250, 204],
        'fbs': [1, 0, 0, 0, 0], 'restecg': [2, 2, 2, 0, 2], 'thalach': [150, 108, 129, 187, 172],
        'exang': [0, 1, 1, 0, 0], 'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4], 'slope': [3, 2, 2, 3, 1],
        'ca': ['0.0', '3.0', '2.0', '?', '0.0'], # '?' simulates missing data
        'thal': ['6.0', '3.0', '7.0', '3.0', '?'], # '?' simulates missing data
        'target': [0, 2, 1, 0, 4] # Values > 0 mean disease is present
    })
    file_path = tmp_path / "dummy_heart.csv"
    df.to_csv(file_path, index=False, header=False)
    return str(file_path)

@pytest.fixture
def real_data_path():
    return "data/raw/heart.csv"

# ==========================================
# 1. UNIT TESTS FOR PREPROCESSING (6 required)
# ==========================================
def test_handles_missing_values(sample_data_path):
    df = load_and_clean_data(sample_data_path)
    assert df['ca'].isna().sum() == 0  # Assert no missing values left
    assert df['thal'].isna().sum() == 0

def test_binarizes_target_variable(sample_data_path):
    df = load_and_clean_data(sample_data_path)
    assert set(df['target'].unique()).issubset({0, 1}) # Assert only 0 or 1 exists

def test_split_data_shapes(sample_data_path):
    df = load_and_clean_data(sample_data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.4, random_state=42)
    assert len(X_train) == 3
    assert len(X_test) == 2

def test_does_not_modify_original_dataframe(sample_data_path):
    df = load_and_clean_data(sample_data_path)
    original_shape = df.shape
    split_data(df, test_size=0.2, random_state=42)
    assert df.shape == original_shape # Assert the split didn't delete rows from original

def test_invalid_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_and_clean_data("fake_directory/non_existent_file.csv")

def test_invalid_test_size_raises_error(sample_data_path):
    df = load_and_clean_data(sample_data_path)
    with pytest.raises(ValueError):
        split_data(df, test_size=1.5, random_state=42) # 1.5 is an invalid split ratio

# ==========================================
# 2. DATA VALIDATION TESTS (3 required)
# ==========================================
def test_expected_columns_are_present(real_data_path):
    df = load_and_clean_data(real_data_path)
    expected_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    assert list(df.columns) == expected_cols

def test_target_values_are_valid(real_data_path):
    df = load_and_clean_data(real_data_path)
    assert df['target'].isin([0, 1]).all()

def test_numeric_ranges_are_physiological(real_data_path):
    df = load_and_clean_data(real_data_path)
    assert df['age'].between(0, 120).all() # Age must be realistic
    assert (df['chol'] > 0).all() # Cholesterol must be positive

# ==========================================
# 3. MODEL VALIDATION TESTS (2 required)
# ==========================================
def test_model_prediction_type_and_shape(sample_data_path):
    df = load_and_clean_data(sample_data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.4, random_state=42)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    assert len(preds) == len(y_test) # Output shape matches input
    assert set(preds).issubset({0, 1}) # Output is binary classification

def test_model_minimum_accuracy(real_data_path):
    df = load_and_clean_data(real_data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, preds)
    assert accuracy > 0.70 # Assert model performs better than random guessing