import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TradingModel:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None

    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare stock data for training

        Args:
            data (pd.DataFrame): Raw stock data with OHLCV columns

        Returns:
            tuple: Processed X and y data for training
        """
        # Calculate technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # Create features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI']
        X = data[features].values
        
        # Create labels (1 for price increase, 0 for decrease)
        y = (data['Close'].shift(-1) > data['Close']).astype(int).values
        
        # Remove rows with NaN
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y

    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index

        Args:
            prices (pd.Series): Price data
            periods (int, optional): RSI period. Defaults to 14.

        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))