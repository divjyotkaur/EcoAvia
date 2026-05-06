"""
LSTM Worker Module (scikit-learn MLPRegressor)
Trains a neural network on ARIMA residuals for non-linear pattern capture.
Uses scikit-learn MLPRegressor instead of PyTorch for broader compatibility.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import config


class LSTMWorker:
    """Neural Network regressor for capturing non-linear patterns in residuals."""

    def __init__(self, input_size, hidden_sizes=None, lookback=12, dropout=0.2):
        if hidden_sizes is None:
            hidden_sizes = config.LSTM_HIDDEN_UNITS

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.lookback = lookback
        self.dropout = dropout
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False

    def _make_sequences(self, X, y):
        """
        Create sliding window sequences.
        Args:
            X: (n_samples, n_features)
            y: (n_samples,)

        Returns:
            X_seq: (n_samples - lookback, lookback * n_features)
            y_seq: (n_samples - lookback,)
        """
        X_seq = []
        y_seq = []
        for i in range(self.lookback, len(X)):
            # Flatten the lookback window
            window = X[i - self.lookback:i, :].flatten()
            X_seq.append(window)
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit neural network on training data.

        Args:
            X_train: (n_train, n_features) - normalized features
            y_train: (n_train,) - passenger demand
            X_val: validation features (optional)
            y_val: validation targets (optional)

        Returns:
            self
        """
        # Create sequences
        X_seq, y_seq = self._make_sequences(X_train, y_train)

        # Scale features
        X_seq_scaled = self.scaler.fit_transform(X_seq)

        # Create MLPRegressor with hidden layer sizes
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_sizes),
            activation='relu',
            solver='adam',
            learning_rate_init=config.LSTM_LEARNING_RATE,
            max_iter=config.LSTM_EPOCHS,
            early_stopping=True if X_val is not None else False,
            validation_fraction=0.1 if X_val is not None else 0.0,
            n_iter_no_change=config.LSTM_EARLY_STOPPING_PATIENCE,
            random_state=42,
            verbose=0
        )

        # Fit the model
        self.model.fit(X_seq_scaled, y_seq)
        self.fitted = True

        print(f"[LSTM] Trained neural network with hidden layers: {self.hidden_sizes}")
        return self

    def predict(self, X_test, use_mc_dropout=False, mc_samples=50):
        """
        Make predictions on test data.

        Args:
            X_test: (n_test, n_features)
            use_mc_dropout: ignored (sklearn doesn't support MC dropout)
            mc_samples: ignored

        Returns:
            predictions: np.array of shape (n_test,) - padded to match input length
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_seq, _ = self._make_sequences(X_test, np.zeros(len(X_test)))
        X_seq_scaled = self.scaler.transform(X_seq)

        predictions = self.model.predict(X_seq_scaled)

        # Pad predictions with mean value to match input length
        if len(predictions) < len(X_test):
            pad_size = len(X_test) - len(predictions)
            padding = np.full(pad_size, np.mean(predictions))
            predictions = np.concatenate([padding, predictions])

        return predictions

    def save(self, path):
        """Save model checkpoint."""
        import joblib
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        print(f"[OK] LSTM model saved to {path}")

    def load(self, path):
        """Load model checkpoint."""
        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.fitted = True
        print(f"[OK] LSTM model loaded from {path}")
