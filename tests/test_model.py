import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.preprocess import load_and_clean_data, split_data

RAW_DATA_PATH = "data/raw/heart.csv"


@pytest.fixture(scope="module")
def trained_model_and_test_data():
    df = load_and_clean_data(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_predictions_are_binary_and_correct_shape(trained_model_and_test_data):
    model, X_test, y_test = trained_model_and_test_data
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})


def test_model_meets_minimum_accuracy_threshold(trained_model_and_test_data):
    model, X_test, y_test = trained_model_and_test_data
    accuracy = accuracy_score(y_test, model.predict(X_test))
    assert accuracy > 0.70, f"Accuracy {accuracy:.3f} is below the 0.70 production threshold"
