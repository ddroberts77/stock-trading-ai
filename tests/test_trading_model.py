import unittest
import numpy as np
import pandas as pd
from src.models.trading_model import TradingModel

class TestTradingModel(unittest.TestCase):
    def setUp(self):
        self.model = TradingModel()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31')
        self.test_data = pd.DataFrame({
            'Open': np.random.rand(len(dates)),
            'High': np.random.rand(len(dates)),
            'Low': np.random.rand(len(dates)),
            'Close': np.random.rand(len(dates)),
            'Volume': np.random.randint(1000, 100000, len(dates))
        }, index=dates)

    def test_prepare_data(self):
        X, y = self.model.prepare_data(self.test_data)
        
        # Check shapes
        self.assertEqual(X.shape[1], 8)  # 8 features
        self.assertEqual(X.shape[0], y.shape[0])  # Same number of samples
        
        # Check scaling
        self.assertTrue(np.all(X >= 0))
        self.assertTrue(np.all(X <= 1))
        
        # Check labels
        self.assertTrue(np.all(np.isin(y, [0, 1])))

    def test_calculate_rsi(self):
        rsi = self.model._calculate_rsi(self.test_data['Close'])
        
        # Check RSI ranges
        self.assertTrue(np.all(rsi[~np.isnan(rsi)] >= 0))
        self.assertTrue(np.all(rsi[~np.isnan(rsi)] <= 100))