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
