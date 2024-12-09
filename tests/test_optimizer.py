import unittest
import torch
import torch.nn as nn
from src.models.model_optimizer import ModelOptimizer

class TestModelOptimizer(unittest.TestCase):
    def setUp(self):
        def create_model(hidden_size=50, learning_rate=0.001):
            model = nn.Sequential(
                nn.LSTM(input_size=5, hidden_size=hidden_size, batch_first=True),
                nn.Linear(hidden_size, 1)
            )
            return model

        self.param_grid = {
            'hidden_size': [32, 64],
            'learning_rate': [0.001, 0.01]
        }
        self.optimizer = ModelOptimizer(create_model, self.param_grid)

    def test_optimization(self):
        # Generate dummy data
        X = torch.randn(100, 10, 5)  # (batch_size, seq_length, features)
        y = torch.randn(100, 1)      # (batch_size, 1)

        # Test optimization
        grid_result = self.optimizer.optimize(X, y, cv=2)
        self.assertIsNotNone(self.optimizer.best_params)

    def test_train_best_model(self):
        X = torch.randn(100, 10, 5)
        y = torch.randn(100, 1)

        # First optimize
        self.optimizer.optimize(X, y, cv=2)

        # Then train
        history = self.optimizer.train_best_model(X, y)
        self.assertIsNotNone(history)

if __name__ == '__main__':
    unittest.main()