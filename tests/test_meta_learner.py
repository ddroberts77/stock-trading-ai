import pytest
import torch
from src.models.meta_learner import MarketMetaLearner

def test_meta_learner_initialization():
    model = MarketMetaLearner(input_size=10, hidden_size=20, num_layers=2)
    assert isinstance(model, MarketMetaLearner)
    assert model.hidden_size == 20
    assert model.num_layers == 2

def test_meta_learner_forward_pass():
    model = MarketMetaLearner(input_size=10, hidden_size=20, num_layers=2)
    batch_size = 32
    seq_length = 50
    x = torch.randn(batch_size, seq_length, 10)
    market_data = torch.randn(batch_size, 5)  # Additional market features
    
    market_regime, adaptation_params = model(x, market_data)
    
    assert market_regime.shape == (batch_size, 3)
    assert adaptation_params.shape == (batch_size, 20)