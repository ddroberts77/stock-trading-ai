import pytest
import torch
from src.models.meta_learner import MarketMetaLearner

@pytest.fixture
def model(device):
    return MarketMetaLearner(input_size=10, hidden_size=20, num_layers=2).to(device)

def test_meta_learner_initialization(model):
    assert isinstance(model, MarketMetaLearner)
    assert model.hidden_size == 20
    assert model.num_layers == 2

def test_meta_learner_forward_pass(model, sample_market_data, sample_market_info):
    with torch.no_grad():
        market_regime, adaptation_params = model(sample_market_data, sample_market_info)
    
    assert market_regime.shape == (16, 3)
    assert adaptation_params.shape == (16, 20)
    assert torch.isfinite(market_regime).all()
    assert torch.isfinite(adaptation_params).all()

def test_meta_learner_backward_pass(model, sample_market_data, sample_market_info):
    model.train()
    
    market_regime, adaptation_params = model(sample_market_data, sample_market_info)
    loss = market_regime.mean() + adaptation_params.mean()
    loss.backward()
    
    # Check gradients exist and are finite
    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()