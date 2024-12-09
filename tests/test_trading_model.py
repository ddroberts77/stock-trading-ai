import pytest
import torch
from src.models.trading_model import TradingModel

@pytest.fixture
def model():
    return TradingModel(
        input_size=10,
        hidden_size=32,
        num_assets=5
    )

def test_trading_model_initialization(model):
    assert isinstance(model, TradingModel)
    assert model.input_size == 10
    assert model.hidden_size == 32
    assert model.num_assets == 5

def test_trading_model_forward_pass(model):
    batch_size = 16
    seq_length = 20
    
    # Create dummy input data
    market_data = torch.randn(batch_size, seq_length, 10)
    
    # Get model predictions
    with torch.no_grad():
        positions = model(market_data)
    
    # Check output shape and constraints
    assert positions.shape == (batch_size, model.num_assets)
    assert torch.all(positions >= -1) and torch.all(positions <= 1)
    
    # Sum of absolute positions should be <= 1 (leverage constraint)
    assert torch.all(torch.abs(positions).sum(dim=1) <= 1.01)  # Allow small numerical error

def test_trading_model_loss_calculation(model):
    batch_size = 8
    
    # Generate dummy positions and returns
    positions = torch.randn(batch_size, model.num_assets)
    asset_returns = torch.randn(batch_size, model.num_assets)
    
    # Calculate portfolio returns
    portfolio_returns = (positions * asset_returns).sum(dim=1)
    
    # Calculate Sharpe ratio loss
    returns_mean = portfolio_returns.mean()
    returns_std = portfolio_returns.std()
    sharpe = returns_mean / (returns_std + 1e-6)
    
    assert isinstance(sharpe.item(), float)
    assert not torch.isnan(sharpe)

def test_trading_model_gradient_flow(model):
    batch_size = 4
    seq_length = 10
    
    # Create inputs that require gradients
    market_data = torch.randn(batch_size, seq_length, 10, requires_grad=True)
    
    # Forward pass
    positions = model(market_data)
    
    # Dummy loss
    loss = positions.mean()
    
    # Test backward pass
    loss.backward()
    
    # Verify gradients exist and are finite
    assert market_data.grad is not None
    assert torch.isfinite(market_data.grad).all()