import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class DataLoader:
    @staticmethod
    def load_stock_data(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load historical stock data from Yahoo Finance

        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format

        Returns:
            pd.DataFrame: Historical stock data
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            return data
        except Exception as e:
            print(f"Error loading data for {symbol}: {str(e)}")
            return None

    @staticmethod
    def save_data(data: pd.DataFrame, filepath: str) -> None:
        """Save data to CSV file

        Args:
            data (pd.DataFrame): Data to save
            filepath (str): Path to save file
        """
        try:
            data.to_csv(filepath)
            print(f"Data saved to {filepath}")
        except Exception as e:
            print(f"Error saving data: {str(e)}")