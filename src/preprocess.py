import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path):
    """Loads data, handles missing values, and binarizes the target."""
    # 1. Define the physiological columns
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    # Load the CSV and apply the headers
    df = pd.read_csv(file_path, names=columns)
    
    # 2. Handle missing values
    # The UCI dataset uses '?' for missing data. 'coerce' safely forces them to NaN.
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
    
    # Fill those new NaNs with the median value of their respective columns
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].median())
    
    # 3. Binarize the target variable (0 = healthy, 1+ = heart disease present)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    return df

def split_data(df, test_size, random_state):
    """Splits the dataframe into training and testing features/targets."""
    X = df.drop('target', axis=1)
    y = df['target']
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)