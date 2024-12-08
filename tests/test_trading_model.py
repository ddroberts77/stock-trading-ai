import pytest
import pandas as pd
import numpy as np
from src.models.trading_model import TradingModel

@pytest.fixture
def sample_data():
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100)
    df = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    return df

def test_model_initialization():
    model = TradingModel()
    assert model is not None

def test_predict_next_movement(sample_data):
    model = TradingModel()
    # Just verify it runs without error for now
    try:
        model.predict_next_movement(sample_data)
        assert True
    except:
        assert False

def test_model_training(sample_data):
    model = TradingModel()
    try:
        model.train(sample_data)
        assert True
    except:
        assert False