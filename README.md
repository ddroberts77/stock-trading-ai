# Stock Trading AI

[![Python Tests](https://github.com/ddroberts77/stock-trading-ai/actions/workflows/python-tests.yml/badge.svg)](https://github.com/ddroberts77/stock-trading-ai/actions/workflows/python-tests.yml)

An AI-powered stock trading system using machine learning for market analysis and trading decisions.

## Features

- Real-time stock data fetching using yfinance
- Technical indicator calculation and analysis
- ML model training and prediction
- Performance validation and backtesting
- Automated trading suggestions

## Project Structure

```
stock-trading-ai/
├── src/
│   ├── data/          # Data handling and preprocessing
│   └── models/        # ML models and trading logic
├── tests/             # Unit and integration tests
├── docs/              # Documentation
└── data/              # Stock data and trained models
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ddroberts77/stock-trading-ai.git
cd stock-trading-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python src/models/trading_model.py --train
```

2. Make predictions:
```bash
python src/models/trading_model.py --predict AAPL
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Push your changes
4. Create a pull request

## License

MIT
