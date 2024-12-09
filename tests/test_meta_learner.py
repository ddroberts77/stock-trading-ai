import pytest
import torch
from src.models.meta_learner import MarketMetaLearner

@pytest.fixture
def model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return MarketMetaLearner(input_size=10, hidden_size=20, num_layers=2).to(device)

@pytest.fixture
def sample_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    seq_length = 20
    feature_dim = 10
    market_features = 5
    x = torch.randn(batch_size, seq_length, feature_dim).to(device)
    market_data = torch.randn(batch_size, market_features).to(device)
    return x, market_data

def test_meta_learner_initialization(model):
    assert isinstance(model, MarketMetaLearner)
    assert model.hidden_size == 20
    assert model.num_layers == 2

def test_meta_learner_forward_pass(model, sample_data):
    x, market_data = sample_data
    with torch.no_grad():
        market_regime, adaptation_params = model(x, market_data)
    
    assert market_regime.shape == (16, 3)
    assert adaptation_params.shape == (16, 20)
    assert torch.isfinite(market_regime).all()
    assert torch.isfinite(adaptation_params).all()