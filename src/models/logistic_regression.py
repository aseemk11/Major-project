# logistic_regression.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def train_logistic_regression(df, target_column):
    """
    Train a logistic regression model on the given dataset.
    
    Args:
    - df (pd.DataFrame): The dataset containing features and the target column.
    - target_column (str): The column name for the target variable.
    
    Returns:
    - model (LogisticRegression): Trained logistic regression model.
    - X_test (pd.DataFrame): Test features for model evaluation.
    - y_test (pd.Series): Test target for model evaluation.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    
    # Print Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy}")
    
    return model, X_test, y_test
