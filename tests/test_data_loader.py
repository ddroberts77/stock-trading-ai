import pytest
from src.data.data_loader import DataLoader

def test_data_loader_initialization():
    loader = DataLoader()
    assert isinstance(loader, DataLoader)