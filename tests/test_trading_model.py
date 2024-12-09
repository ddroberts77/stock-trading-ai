import pytest
import torch
from src.models.trading_model import TradingModel

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def model(device):
    return TradingModel(
        input_size=10,
        hidden_size=32,
        num_assets=5
    ).to(device)

def test_trading_model_initialization(model):
    assert isinstance(model, TradingModel)
    assert model.input_size == 10
    assert model.hidden_size == 32
    assert model.num_assets == 5

def test_trading_model_forward_pass(model, device):
    batch_size = 16
    seq_length = 20
    
    market_data = torch.randn(batch_size, seq_length, 10).to(device)
    
    with torch.no_grad():
        positions = model(market_data)
    
    assert positions.shape == (batch_size, model.num_assets)
    assert torch.all(positions >= -1) and torch.all(positions <= 1)
    assert torch.all(torch.abs(positions).sum(dim=1) <= 1.01)

def test_trading_model_loss_calculation(model, device):
    batch_size = 8
    
    positions = torch.randn(batch_size, model.num_assets).to(device)
    asset_returns = torch.randn(batch_size, model.num_assets).to(device)
    
    portfolio_returns = (positions * asset_returns).sum(dim=1)
    
    returns_mean = portfolio_returns.mean()
    returns_std = portfolio_returns.std()
    sharpe = returns_mean / (returns_std + 1e-6)
    
    assert isinstance(sharpe.item(), float)
    assert not torch.isnan(sharpe)

def test_trading_model_gradient_flow(model, device):
    batch_size = 4
    seq_length = 10
    
    market_data = torch.randn(batch_size, seq_length, 10, requires_grad=True).to(device)
    
    positions = model(market_data)
    loss = positions.mean()
    loss.backward()
    
    assert market_data.grad is not None
    assert torch.isfinite(market_data.grad).all()