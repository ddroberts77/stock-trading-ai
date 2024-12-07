import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

class DataPipeline:
    """Data pipeline for fetching and processing stock market data."""

    def __init__(self):
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.cached_data: Dict[str, pd.DataFrame] = {}

    def fetch_stock_data(self, 
                        symbol: str,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        interval: str = '1d') -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance."""
        try:
            # Default to last year if no dates provided
            if not start_date:
                start_date = datetime.now() - timedelta(days=365)
            if not end_date:
                end_date = datetime.now()

            # Fetch data
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date, interval=interval)

            # Basic validation
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Cache the data
            self.cached_data[symbol] = df

            return df

        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")

    def process_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        df = data.copy()

        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()

        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        return df

    def normalize_features(self, 
                          data: pd.DataFrame,
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Normalize features using Min-Max scaling."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns

        df = data.copy()
        
        for col in columns:
            if col not in self.scalers:
                self.scalers[col] = MinMaxScaler()
                df[col] = self.scalers[col].fit_transform(df[[col]])
            else:
                df[col] = self.scalers[col].transform(df[[col]])

        return df

    def prepare_training_data(self,
                             data: pd.DataFrame,
                             target_column: str = 'Close',
                             sequence_length: int = 10,
                             prediction_horizon: int = 1) -> tuple:
        """Prepare sequences for training."""
        df = data.copy()

        # Calculate target (price movement direction)
        df['Target'] = np.where(
            df[target_column].shift(-prediction_horizon) > df[target_column], 1, 0
        )

        # Create sequences
        sequences = []
        targets = []

        for i in range(len(df) - sequence_length - prediction_horizon + 1):
            seq = df.iloc[i:(i + sequence_length)]
            target = df.iloc[i + sequence_length + prediction_horizon - 1]['Target']
            
            sequences.append(seq.values)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def inverse_transform(self, 
                         data: np.ndarray,
                         column: str) -> np.ndarray:
        """Inverse transform normalized data."""
        if column not in self.scalers:
            raise ValueError(f"No scaler found for column {column}")
            
        return self.scalers[column].inverse_transform(data.reshape(-1, 1))

    def get_latest_data(self, 
                        symbol: str,
                        lookback_days: int = 100) -> pd.DataFrame:
        """Get the most recent data for a symbol."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        return self.fetch_stock_data(symbol, start_date, end_date)