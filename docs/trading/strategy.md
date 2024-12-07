# Trading Strategy Documentation

## Strategy Overview
The trading strategy combines LSTM predictions with risk management rules for optimal trade execution.

## Entry Rules
1. Strong buy signal (prediction > threshold)
2. Confirmation from technical indicators
3. Market trend alignment

## Exit Rules
1. Take profit targets reached
2. Stop loss triggered
3. Signal reversal

## Position Sizing
```python
def calculate_position_size(capital, risk_per_trade, stop_loss):
    risk_amount = capital * risk_per_trade
    position_size = risk_amount / stop_loss
    return position_size
```

## Risk Management
- Maximum position size: 5% of capital
- Stop loss: 2% per trade
- Maximum drawdown: 15%
- Position correlation limits
