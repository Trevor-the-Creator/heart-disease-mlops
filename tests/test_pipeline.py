import pandas as pd
import numpy as np
from src.preprocess import load_and_clean_data

def test_categorical_encoding(tmp_path):
    # Create mock data with a discrete categorical feature 'cp'
    mock_data = pd.DataFrame({
        "age": [50, 60, 45],
        "cp": [1, 2, 3],  # Categorical mapping feature
        "target": [1, 0, 1]
    })
    
    mock_file = tmp_path / "mock_heart.csv"
    mock_data.to_csv(mock_file, index=False)
    
    processed_df = load_and_clean_data(str(mock_file))
    
    # Assert that our get_dummies logic generated encoded flags
    # if 'cp' encoding acts correctly, the original 'cp' column expands or drops
    assert "cp" not in processed_df.columns or any("cp_" in col for col in processed_df.columns)