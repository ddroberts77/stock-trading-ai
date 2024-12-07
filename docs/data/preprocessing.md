# Data Preprocessing Guide

## Data Sources
1. Historical price data (OHLCV)
2. Technical indicators
3. Fundamental data
4. Market sentiment

## Preprocessing Steps

### 1. Data Cleaning
```python
def clean_price_data(df):
    # Remove missing values
    df = df.dropna()
    
    # Handle outliers
    df = remove_outliers(df, columns=['open', 'high', 'low', 'close', 'volume'])
    
    return df
```

### 2. Feature Engineering
```python
def create_features(df):
    # Technical indicators
    df['MA_20'] = df['close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'] = calculate_macd(df['close'])
    
    return df
```

### 3. Data Normalization
```python
from sklearn.preprocessing import MinMaxScaler

def normalize_data(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled, scaler
```
