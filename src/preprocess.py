import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    # Handle structural missing data drops if any string-typed NaNs exist
    df = df.dropna()
    
    # Explicit categorical variable handling via pd.get_dummies
    # This addresses Victor's explicit requirement for categorical encoding support
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    # Intersect with existing columns to avoid key errors if columns are already numeric
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
        
    return df

def split_data(df, test_size=0.2, random_state=42):
    if "target" in df.columns:
        X = df.drop(columns=["target"])
        y = df["target"]
    else:
        # Fallback if target column name differs slightly
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
    return train_test_split(X, y, test_size=test_size, random_state=random_state)