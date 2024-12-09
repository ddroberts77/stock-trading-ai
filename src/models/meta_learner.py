import torch
import torch.nn as nn

class MarketMetaLearner(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Feature extraction
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Market regime classifier
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_size + 5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )
        
        # Adaptation parameters generator
        self.adaptation_head = nn.Sequential(
            nn.Linear(hidden_size + 5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, x: torch.Tensor, market_data: torch.Tensor):
        lstm_out, (h_n, _) = self.lstm(x)
        final_hidden = h_n[-1]
        
        combined = torch.cat([final_hidden, market_data], dim=1)
        
        market_regime = self.regime_head(combined)
        adaptation_params = self.adaptation_head(combined)
        
        return market_regime, adaptation_params