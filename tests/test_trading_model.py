import unittest
import pandas as pd
import numpy as np
from src.models.trading_model import TradingModel

class TestTradingModel(unittest.TestCase):
    def setUp(self):
        self.model = TradingModel()
        
        # Create sample data
        self.test_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=100),
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        })
        self.test_data.set_index('Date', inplace=True)

    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model)

    def test_predict_next_movement(self):
        """Test prediction of next price movement"""
        prediction = self.model.predict_next_movement(self.test_data)
        self.assertIsNotNone(prediction)
        self.assertIn(prediction, ['up', 'down', 'hold'])

    def test_train_model(self):
        """Test model training"""
        training_result = self.model.train(self.test_data)
        self.assertTrue(training_result)  # Training should return True on success

    def test_model_evaluation(self):
        """Test model evaluation metrics"""
        # Train the model first
        self.model.train(self.test_data)
        
        # Get evaluation metrics
        metrics = self.model.evaluate(self.test_data)
        
        # Check metrics exist
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        
        # Check metrics are in valid ranges
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
