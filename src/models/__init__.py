# src/models/__init__.py

from .train_model import train_logistic_regression
from .evaluate_model import evaluate_model_function

__all__ = ["train_logistic_regression", "evaluate_model_function"]
