import yfinance as yf
import pandas as pd
from typing import List, Dict, Union

class MarketDataLoader:
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_stock_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for given symbols."""
        data = {}
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            data[symbol] = ticker.history(start=self.start_date, end=self.end_date)
        return data
    
    def get_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        return df
    
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))