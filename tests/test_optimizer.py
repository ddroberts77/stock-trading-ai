import pytest
import torch
import torch.nn as nn
from src.models.model_optimizer import ModelOptimizer

class SimpleModel(nn.Module):
    def __init__(self, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])  # Use last timestep

@pytest.fixture
def model_creator():
    def create_model(hidden_size=50, learning_rate=0.001):
        return SimpleModel(hidden_size=hidden_size)
    return create_model

@pytest.fixture
def param_grid():
    return {
        'hidden_size': [32, 64],
        'learning_rate': [0.001, 0.01]
    }

@pytest.fixture
def optimizer(model_creator, param_grid):
    return ModelOptimizer(model_creator, param_grid)

def test_optimization(optimizer):
    X = torch.randn(100, 10, 5)  # (batch_size, seq_length, features)
    y = torch.randn(100, 1)      # (batch_size, 1)
    
    grid_result = optimizer.optimize(X, y, cv=2)
    assert optimizer.best_params is not None
    assert 'hidden_size' in optimizer.best_params
    assert 'learning_rate' in optimizer.best_params

def test_train_best_model(optimizer):
    X = torch.randn(100, 10, 5)
    y = torch.randn(100, 1)
    
    # First optimize
    optimizer.optimize(X, y, cv=2)
    
    # Then train
    history = optimizer.train_best_model(X, y)
    assert 'loss' in history
    assert len(history['loss']) > 0
    assert all(isinstance(loss, float) for loss in history['loss'])
    assert all(loss >= 0 for loss in history['loss'])