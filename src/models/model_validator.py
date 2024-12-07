import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

class ModelValidator:
    def __init__(self, model):
        self.model = model
        self.metrics = {}

    def calculate_metrics(self, X_test, y_test):
        """Calculate various performance metrics"""
        predictions = self.model.predict(X_test)
        self.metrics['mse'] = mean_squared_error(y_test, predictions)
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        self.metrics['mae'] = mean_absolute_error(y_test, predictions)
        return self.metrics

    def generate_report(self):
        """Generate performance report"""
        if not self.metrics:
            raise ValueError('Must run calculate_metrics() first')

        report = pd.DataFrame({
            'Metric': list(self.metrics.keys()),
            'Value': list(self.metrics.values())
        })
        return report

    def validate_predictions(self, X_test, y_test, threshold=0.1):
        """Validate prediction accuracy"""
        predictions = self.model.predict(X_test)
        errors = np.abs(predictions - y_test) / y_test
        accuracy = np.mean(errors <= threshold)
        return accuracy