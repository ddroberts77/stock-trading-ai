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
    
    # Test regime probabilities sum close to 1
    regime_probs = torch.softmax(market_regime, dim=1)
    assert torch.allclose(regime_probs.sum(dim=1), torch.ones(16).to(regime_probs.device))

def test_meta_learner_backward_pass(model, sample_market_data, sample_market_info):
    model.train()
    
    market_regime, adaptation_params = model(sample_market_data, sample_market_info)
    loss = market_regime.mean() + adaptation_params.mean()
    loss.backward()
    
    # Check gradients exist and are finite
    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()

def test_meta_learner_regime_output_range(model, sample_market_data, sample_market_info):
    with torch.no_grad():
        market_regime, _ = model(sample_market_data, sample_market_info)
        regime_probs = torch.softmax(market_regime, dim=1)
    
    assert torch.all(regime_probs >= 0) and torch.all(regime_probs <= 1)
    assert torch.allclose(regime_probs.sum(dim=1), torch.ones(16).to(regime_probs.device))

def test_meta_learner_different_batch_sizes(model):
    batch_sizes = [1, 4, 32]
    seq_length = 20
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, seq_length, 10).to(model.lstm.weight_ih_l0.device)
        market_data = torch.randn(batch_size, 5).to(model.lstm.weight_ih_l0.device)
        
        with torch.no_grad():
            market_regime, adaptation_params = model(x, market_data)
        
        assert market_regime.shape == (batch_size, 3)
        assert adaptation_params.shape == (batch_size, 20)