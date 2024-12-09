import pytest
import torch

@pytest.fixture(scope='session')
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(42)
    return None