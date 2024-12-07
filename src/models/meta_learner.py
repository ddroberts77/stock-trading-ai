import torch
import torch.nn as nn
from typing import Tuple, Dict

class MarketMetaLearner(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(MarketMetaLearner, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Feature extraction
        self.feature_extractor = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        
        # Market regime classification
        self.market_classifier = nn.Linear(hidden_size, 3)  # 3 market regimes
        
        # Adaptation network
        self.adaptation_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, x: torch.Tensor, market_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract features
        features, _ = self.feature_extractor(x)
        
        # Classify market regime
        market_regime = self.market_classifier(features[:, -1, :])
        
        # Generate adaptation parameters
        adaptation_params = self.adaptation_network(features[:, -1, :])
        
        return market_regime, adaptation_params
    
    def adapt_to_market(self, market_regime: torch.Tensor, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Adapt model parameters based on market regime."""
        adapted_params = {}
        for name, param in self.named_parameters():
            if 'feature_extractor' in name:
                adapted_params[name] = param + params * market_regime.unsqueeze(-1)
        return adapted_params