import unittest
import numpy as np
from src.models.model_optimizer import ModelOptimizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

class TestModelOptimizer(unittest.TestCase):
    def setUp(self):
        def create_model(units=50, learning_rate=0.001):
            model = Sequential([
                LSTM(units=units, return_sequences=True),
                LSTM(units=units//2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            return model

        self.param_grid = {
            'units': [32, 64],
            'learning_rate': [0.001, 0.01]
        }
        self.optimizer = ModelOptimizer(create_model, self.param_grid)

    def test_optimization(self):
        # Generate dummy data
        X = np.random.random((100, 10, 5))
        y = np.random.random(100)

        # Test optimization
        grid_result = self.optimizer.optimize(X, y, cv=2)
        self.assertIsNotNone(self.optimizer.best_params)

    def test_train_best_model(self):
        X = np.random.random((100, 10, 5))
        y = np.random.random(100)

        # First optimize
        self.optimizer.optimize(X, y, cv=2)

        # Then train
        history = self.optimizer.train_best_model(X, y)
        self.assertIsNotNone(history)

if __name__ == '__main__':
    unittest.main()