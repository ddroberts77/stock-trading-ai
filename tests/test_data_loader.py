import pytest
import pandas as pd
from src.data.stock_data_loader import StockDataLoader

@pytest.fixture
def data_loader():
    return StockDataLoader()

def test_data_loader_initialization(data_loader):
    assert data_loader is not None
    assert hasattr(data_loader, 'fetch_stock_data')

def test_fetch_stock_data_handles_errors(data_loader):
    # Test with invalid symbol
    result = data_loader.fetch_stock_data('INVALID_SYMBOL_123')
    assert result is None