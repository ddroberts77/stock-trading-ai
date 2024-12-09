import pytest
import torch
from src.models.trading_model import TradingModel

@pytest.fixture
def model(device):
    return TradingModel(input_size=10, hidden_size=32, num_assets=5).to(device)

def test_trading_model_initialization(model):
    assert isinstance(model, TradingModel)
    assert model.input_size == 10
    assert model.hidden_size == 32
    assert model.num_assets == 5

def test_trading_model_forward_pass(model, sample_market_data):
    with torch.no_grad():
        positions = model(sample_market_data)
    
    assert positions.shape == (16, 5)
    assert torch.all(positions >= -1) and torch.all(positions <= 1)
    assert torch.all(torch.abs(positions).sum(dim=1) <= 1.01)
