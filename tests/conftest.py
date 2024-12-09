import pytest
import torch

@pytest.fixture(scope='session')
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(42)
    return None

@pytest.fixture
def sample_market_data(device):
    batch_size = 16
    seq_length = 20
    feature_dim = 10
    return torch.randn(batch_size, seq_length, feature_dim).to(device)

@pytest.fixture
def sample_market_info(device):
    batch_size = 16
    market_features = 5
    return torch.randn(batch_size, market_features).to(device)