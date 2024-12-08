import pytest
from src.models.trading_model import TradingModel

def test_trading_model_initialization():
    model = TradingModel()
    assert isinstance(model, TradingModel)