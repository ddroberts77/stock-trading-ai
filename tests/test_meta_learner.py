import pytest
import torch
from src.models.meta_learner import MarketMetaLearner

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_meta_learner_initialization(device):
    model = MarketMetaLearner(input_size=10, hidden_size=20, num_layers=2).to(device)
    assert isinstance(model, MarketMetaLearner)
    assert model.hidden_size == 20
    assert model.num_layers == 2

def test_meta_learner_forward_pass(device):
    model = MarketMetaLearner(input_size=10, hidden_size=20, num_layers=2).to(device)
    batch_size = 32
    seq_length = 50
    
    x = torch.randn(batch_size, seq_length, 10).to(device)
    market_data = torch.randn(batch_size, 5).to(device)
    
    with torch.no_grad():
        market_regime, adaptation_params = model(x, market_data)
    
    assert market_regime.shape == (batch_size, 3)
    assert adaptation_params.shape == (batch_size, 20)
    assert torch.isfinite(market_regime).all()
    assert torch.isfinite(adaptation_params).all()

def test_meta_learner_backward_pass(device):
    model = MarketMetaLearner(input_size=10, hidden_size=20, num_layers=2).to(device)
    batch_size = 8
    seq_length = 10
    
    x = torch.randn(batch_size, seq_length, 10, requires_grad=True).to(device)
    market_data = torch.randn(batch_size, 5, requires_grad=True).to(device)
    
    market_regime, adaptation_params = model(x, market_data)
    loss = market_regime.mean() + adaptation_params.mean()
    loss.backward()
    
    assert x.grad is not None
    assert market_data.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(market_data.grad).all()