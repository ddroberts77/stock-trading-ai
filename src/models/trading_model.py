import torch
import torch.nn as nn

class TradingModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_assets: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_assets = num_assets
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        self.position_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_assets),
            nn.Tanh()
        )
    
    def forward(self, market_data: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, _) = self.lstm(market_data)
        final_hidden = h_n[-1]
        
        positions = self.position_head(final_hidden)
        
        # Apply leverage constraint
        abs_positions = torch.abs(positions)
        scaling_factors = torch.clamp(abs_positions.sum(dim=1), min=1).unsqueeze(1)
        positions = positions / scaling_factors
        
        return positions