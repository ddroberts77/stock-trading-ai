from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Dict, List, Union

class BaseModel(ABC, BaseEstimator):
    """Base class for all trading models."""
    
    def __init__(self):
        self.is_fitted = False
        self.model_params = {}
        self.feature_importance = {}

    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw data into model features."""
        pass

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on given data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data structure and contents."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics."""
        metrics = {
            'accuracy': np.mean(y_true == y_pred),
            'precision': np.sum((y_true == y_pred) & (y_pred == 1)) / np.sum(y_pred == 1),
            'recall': np.sum((y_true == y_pred) & (y_pred == 1)) / np.sum(y_true == 1)
        }
        return metrics

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.feature_importance

    def save_model(self, path: str) -> None:
        """Save model to disk."""
        raise NotImplementedError

    def load_model(self, path: str) -> None:
        """Load model from disk."""
        raise NotImplementedError