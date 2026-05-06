"""
Meta-Learner Module
Paper formula: Yt = α*Yt_ARIMA + β*Yt_LSTM + γ*Yt_XGB, where α+β+γ=1
Learns optimal weights via constrained least squares on validation set.
"""
import numpy as np
from scipy.optimize import minimize
import config


class MetaLearner:
    """Constrained ensemble optimizer."""

    def __init__(self):
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.fitted = False

    def fit(self, y_arima_val, y_lstm_val, y_xgb_val, y_true_val):
        """
        Learn optimal weights on validation set.

        Args:
            y_arima_val: (n_val,) ARIMA predictions
            y_lstm_val: (n_val,) LSTM predictions
            y_xgb_val: (n_val,) XGBoost predictions
            y_true_val: (n_val,) true targets

        Returns:
            self
        """
        # Stack predictions
        predictions = np.column_stack([y_arima_val, y_lstm_val, y_xgb_val])

        # Objective: minimize MSE
        def objective(weights):
            combined = predictions @ weights
            mse = np.mean((combined - y_true_val) ** 2)
            return mse

        # Constraints: α + β + γ = 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # Bounds: all weights in [0, 1]
        bounds = [(0, 1), (0, 1), (0, 1)]

        # Initial guess: equal weights
        x0 = np.array([1/3, 1/3, 1/3])

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-6}
        )

        if result.success:
            self.alpha, self.beta, self.gamma = result.x
            self.fitted = True
            print(f"[MetaLearner] Weights learned: α={self.alpha:.4f}, β={self.beta:.4f}, γ={self.gamma:.4f}")
        else:
            print("[WARNING] Optimization failed, using equal weights")
            self.alpha = self.beta = self.gamma = 1/3
            self.fitted = True

        return self

    def predict(self, y_arima, y_lstm, y_xgb):
        """
        Combine predictions using learned weights.

        Args:
            y_arima: ARIMA predictions
            y_lstm: LSTM predictions
            y_xgb: XGBoost predictions

        Returns:
            combined predictions
        """
        if not self.fitted:
            raise RuntimeError("Weights not learned. Call fit() first.")

        combined = (self.alpha * y_arima +
                    self.beta * y_lstm +
                    self.gamma * y_xgb)
        return combined

    def get_weights(self):
        """Return learned weights as dict."""
        if not self.fitted:
            raise RuntimeError("Weights not learned.")
        return {
            'alpha': float(self.alpha),
            'beta': float(self.beta),
            'gamma': float(self.gamma)
        }


if __name__ == "__main__":
    # Test standalone
    np.random.seed(42)

    # Simulate predictions
    n_val = 24
    y_true = np.random.normal(3.5, 0.3, n_val)
    y_arima = y_true + np.random.normal(0, 0.4, n_val)  # Worse
    y_lstm = y_true + np.random.normal(0, 0.15, n_val)  # Better
    y_xgb = y_true + np.random.normal(0, 0.2, n_val)  # Medium

    # Fit meta-learner
    ml = MetaLearner()
    ml.fit(y_arima, y_lstm, y_xgb, y_true)

    # Combine
    y_combined = ml.predict(y_arima, y_lstm, y_xgb)

    # Evaluate
    mse_arima = np.mean((y_arima - y_true) ** 2)
    mse_lstm = np.mean((y_lstm - y_true) ** 2)
    mse_xgb = np.mean((y_xgb - y_true) ** 2)
    mse_combined = np.mean((y_combined - y_true) ** 2)

    print(f"MSE - ARIMA: {mse_arima:.6f}")
    print(f"MSE - LSTM: {mse_lstm:.6f}")
    print(f"MSE - XGBoost: {mse_xgb:.6f}")
    print(f"MSE - Combined: {mse_combined:.6f}")
    print(f"Weights: {ml.get_weights()}")
