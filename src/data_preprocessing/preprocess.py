# preprocess.py
import pandas as pd

def clean_data(df):
    # Remove missing values, outliers, and erroneous entries
    df = df.dropna()
    # Example outlier removal based on threshold (transaction amount)
    df = df[df['transaction_amount'] < 10000]
    return df

def normalize_data(df):
    # Normalize numeric columns like 'transaction_amount'
    df['transaction_amount'] = (df['transaction_amount'] - df['transaction_amount'].mean()) / df['transaction_amount'].std()
    return df
