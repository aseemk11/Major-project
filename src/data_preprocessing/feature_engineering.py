# feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def create_transaction_features(df):
    """
    Create additional features based on existing columns.
    Example: extracting transaction day, month, year from timestamp.
    
    Args:
    - df (pd.DataFrame): Input DataFrame containing transaction data.
    
    Returns:
    - pd.DataFrame: DataFrame with newly added features.
    """
    df['transaction_month'] = pd.to_datetime(df['transaction_date']).dt.month
    df['transaction_day'] = pd.to_datetime(df['transaction_date']).dt.day
    df['transaction_year'] = pd.to_datetime(df['transaction_date']).dt.year
    df['transaction_hour'] = pd.to_datetime(df['transaction_date']).dt.hour
    df['transaction_weekday'] = pd.to_datetime(df['transaction_date']).dt.weekday

    return df

def encode_categorical_features(df):
    """
    Encode categorical columns using Label Encoding or One-Hot Encoding.
    
    Args:
    - df (pd.DataFrame): Input DataFrame with categorical features.
    
    Returns:
    - pd.DataFrame: DataFrame with encoded features.
    """
    # Example: Label encoding for the 'transaction_type' column
    label_encoder = LabelEncoder()
    df['transaction_type_encoded'] = label_encoder.fit_transform(df['transaction_type'])
    
    # Optional: One-hot encoding for other categorical columns like 'location'
    df = pd.get_dummies(df, columns=['location'], drop_first=True)

    return df

def scale_numeric_features(df):
    """
    Scale numeric features using StandardScaler.
    
    Args:
    - df (pd.DataFrame): Input DataFrame with numeric features to be scaled.
    
    Returns:
    - pd.DataFrame: DataFrame with scaled numeric features.
    """
    scaler = StandardScaler()
    numeric_columns = ['transaction_amount', 'account_age']  # Adjust based on your dataset
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df

def feature_engineering_pipeline(df):
    """
    A full feature engineering pipeline combining all feature engineering functions.
    
    Args:
    - df (pd.DataFrame): Input DataFrame with raw transaction data.
    
    Returns:
    - pd.DataFrame: DataFrame with engineered features.
    """
    df = create_transaction_features(df)
    df = encode_categorical_features(df)
    df = scale_numeric_features(df)

    return df
