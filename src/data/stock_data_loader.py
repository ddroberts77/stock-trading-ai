import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json

class StockDataLoader:
    def __init__(self):
        self.data_path = 'data/stocks/'
    
    def fetch_stock_data(self, symbol, period='1y'):
        """Fetch stock data from Yahoo Finance
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            period (str): Time period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)
        Returns:
            pd.DataFrame: Stock data
        """
        try:
            # Get stock data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            # Add basic info
            info = stock.info
            company_name = info.get('longName', symbol)
            current_price = info.get('currentPrice', df['Close'].iloc[-1])
            
            # Calculate metrics
            df['Daily_Return'] = df['Close'].pct_change()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Save data
            self.save_data(symbol, df, {
                'symbol': symbol,
                'company_name': company_name,
                'current_price': current_price,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def save_data(self, symbol, df, metadata):
        """Save stock data and metadata"""
        # Save CSV
        df.to_csv(f'{self.data_path}{symbol}_data.csv')
        
        # Save metadata
        with open(f'{self.data_path}{symbol}_meta.json', 'w') as f:
            json.dump(metadata, f)

if __name__ == '__main__':
    # Example usage
    loader = StockDataLoader()
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    for symbol in symbols:
        data = loader.fetch_stock_data(symbol)
        if data is not None:
            print(f"Fetched data for {symbol}")