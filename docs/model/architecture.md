# LSTM Model Architecture

## Overview
The stock trading AI system uses a deep learning LSTM (Long Short-Term Memory) architecture for price prediction and trading decisions.

## Model Structure
```python
class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions
```

## Hyperparameters
- Input dimension: 5 (OHLCV data)
- Hidden dimension: 128
- Number of layers: 2
- Learning rate: 0.001
- Batch size: 32
- Sequence length: 60 days
