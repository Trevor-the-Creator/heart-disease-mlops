import json
import pytest
from unittest.mock import MagicMock, patch
import src.interface_utils as iutils


def _make_mock_client(response_payload: dict) -> MagicMock:
    """Build a mock OpenAI client that returns response_payload as JSON text."""
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(response_payload)
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=mock_msg)
    ]
    return mock_client


# ---------------------------------------------------------------------------
# parse_features tests
# ---------------------------------------------------------------------------

def test_parse_features_extracts_complete_feature_set():
    """LLM correctly extracts all 13 features from a detailed description."""
    expected = {
        "age": 55, "sex": 1, "cp": 4, "trestbps": 140, "chol": 240,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3,
    }
    mock_client = _make_mock_client(expected)
    # Patch the lazy client() function to return our mock
    with patch.object(iutils, "client", return_value=mock_client):
        result = iutils.parse_features(
            "55-year-old male, asymptomatic chest pain, BP 140, chol 240, "
            "no high fasting sugar, normal ECG, max HR 150, no exercise angina, "
            "ST depression 1.5, flat slope, 0 vessels, normal thal"
        )
    assert result["age"] == 55
    assert result["sex"] == 1
    assert result["chol"] == 240
    assert result["thal"] == 3


def test_parse_features_returns_null_for_missing_fields():
    """LLM returns null for any feature not present in the user's message."""
    partial = {k: None for k in iutils.REQUIRED_FEATURES}
    partial["age"] = 42
    partial["sex"] = 0

    mock_client = _make_mock_client(partial)
    with patch.object(iutils, "client", return_value=mock_client):
        result = iutils.parse_features("I'm a 42-year-old woman")

    assert result["age"] == 42
    assert result["sex"] == 0
    assert result["cp"] is None
    assert result["chol"] is None


# ---------------------------------------------------------------------------
# prepare_input tests
# ---------------------------------------------------------------------------

def test_prepare_input_produces_single_row_dataframe():
    """prepare_input builds a valid one-row DataFrame with encoded categoricals."""
    features = {
        "age": 55, "sex": 1, "cp": 4, "trestbps": 140, "chol": 240,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3,
    }
    df = iutils.prepare_input(features)
    assert len(df) == 1
    assert "age" in df.columns
    # Categorical columns must be one-hot encoded
    assert "cp" not in df.columns
    assert "thal" not in df.columns
