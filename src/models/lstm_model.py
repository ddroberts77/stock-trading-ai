import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from .base_model import BaseModel

class LSTMModel(BaseModel):
    """LSTM model for time series prediction."""

    def __init__(self, sequence_length: int = 10, n_features: int = 5, n_units: List[int] = [50, 50]):
        super().__init__()
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_units = n_units
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        """Build LSTM model architecture."""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(self.n_units[0],
                      input_shape=(self.sequence_length, self.n_features),
                      return_sequences=len(self.n_units) > 1))
        model.add(Dropout(0.2))

        # Additional LSTM layers
        for i in range(1, len(self.n_units)):
            model.add(LSTM(self.n_units[i],
                          return_sequences=i < len(self.n_units)-1))
            model.add(Dropout(0.2))

        # Output layer
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for LSTM input."""
        # Validate data
        if not self.validate_data(data):
            raise ValueError("Invalid data format")

        # Calculate technical indicators
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['rsi_14'] = self._calculate_rsi(data['close'], 14)
        data['atr_14'] = self._calculate_atr(data[['high', 'low', 'close']], 14)

        # Normalize features
        features = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'rsi_14', 'atr_14']
        data[features] = (data[features] - data[features].mean()) / data[features].std()

        return data.dropna()

    def _prepare_sequences(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare sequential data for LSTM."""
        sequences = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data.iloc[i:(i + self.sequence_length)].values)
        return np.array(sequences)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train LSTM model."""
        X_seq = self._prepare_sequences(X)
        y_seq = y[self.sequence_length:].values

        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )]
        )

        self.is_fitted = True
        self.model_params = history.history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        X_seq = self._prepare_sequences(X)
        predictions = self.model.predict(X_seq)
        return (predictions > 0.5).astype(int)

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()