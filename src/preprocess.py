import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path):
    # 1. Load data, explicitly telling Pandas that '?' means missing data (NaN)
    df = pd.read_csv(file_path, na_values=["?"])
    
    # 2. Check if headers are missing
    if 'age' not in df.columns:
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = pd.read_csv(file_path, na_values=["?"], names=columns)
        
    # 3. Drop rows with missing values
    df = df.dropna()
    
    # 4. Binarize the target variable (0 = no disease, 1-4 = disease present)
    if 'target' in df.columns:
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # 5. Explicit categorical variable encoding
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
        
    return df

def split_data(df, test_size=0.2, random_state=42):
    if "target" in df.columns:
        X = df.drop(columns=["target"])
        y = df["target"]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
    return train_test_split(X, y, test_size=test_size, random_state=random_state)