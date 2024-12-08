import unittest
import pandas as pd
from src.data.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()

    def test_load_stock_data(self):
        # Test with valid symbol
        data = self.loader.load_stock_data('AAPL', '2023-01-01', '2023-12-31')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        
        # Test with invalid symbol
        data = self.loader.load_stock_data('INVALID', '2023-01-01', '2023-12-31')
        self.assertIsNone(data)

    def test_save_data(self):
        # Create test data
        test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        # Test saving to temp file
        self.loader.save_data(test_data, 'test_data.csv')
        
        # Verify file was created
        loaded_data = pd.read_csv('test_data.csv', index_col=0)
        pd.testing.assert_frame_equal(test_data, loaded_data)