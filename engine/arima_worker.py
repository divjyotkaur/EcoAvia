"""
ARIMA Worker Module
Implements SARIMAX(1,1,1)(1,1,1,12) for trend + seasonality decomposition.
Residuals are passed to LSTM per the paper's cascade architecture.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import config


class ARIMAWorker:
    """SARIMAX forecaster for trend and seasonality."""

    def __init__(self):
        self.model = None
        self.result = None
        self.fitted = False

    def fit(self, series, exog=None):
        """
        Fit SARIMAX model on training data.

        Args:
            series: pd.Series of passenger demand (training set)
            exog: DataFrame of exogenous variables (GDP growth, fuel change, COVID dummy)

        Returns:
            self
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        # Fit SARIMAX(1,1,1)(1,1,1,12)
        # p, d, q = 1, 1, 1 (AR, I, MA orders)
        # P, D, Q, s = 1, 1, 1, 12 (Seasonal orders, s=12 for monthly)
        self.model = SARIMAX(
            series,
            order=config.ARIMA_ORDER,
            seasonal_order=config.ARIMA_SEASONAL_ORDER,
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        # Fit with maximum iterations
        self.result = self.model.fit(disp=False, maxiter=200)
        self.fitted = True

        return self

    def predict(self, steps, exog_future=None):
        """
        Generate forecast for next 'steps' periods.

        Args:
            steps: number of periods to forecast
            exog_future: future exogenous variables for forecast period

        Returns:
            forecast: np.array of predicted values
            residuals: np.array of in-sample residuals
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get forecast
        forecast = self.result.get_forecast(steps=steps, exog=exog_future)
        forecast_values = forecast.predicted_mean.values

        # Get in-sample residuals
        residuals = self.result.resid.values

        return forecast_values, residuals

    def forecast_with_ci(self, steps, exog_future=None, alpha=0.05):
        """
        Generate forecast with confidence intervals.

        Args:
            steps: number of periods to forecast
            exog_future: future exogenous variables
            alpha: significance level (0.05 = 95% CI)

        Returns:
            dict with 'forecast', 'ci_lower', 'ci_upper'
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        forecast_result = self.result.get_forecast(steps=steps, exog=exog_future)
        forecast_values = forecast_result.predicted_mean.values
        forecast_ci = forecast_result.conf_int(alpha=alpha)

        return {
            'forecast': forecast_values,
            'ci_lower': forecast_ci.iloc[:, 0].values,
            'ci_upper': forecast_ci.iloc[:, 1].values,
        }

    def get_decomposition(self, series):
        """
        Get seasonal decomposition (for analysis/visualization).
        Xt = Tt + St + Rt (trend + seasonality + residual)
        """
        if isinstance(series, np.ndarray):
            series = pd.Series(series)

        decomp = seasonal_decompose(series, model='additive', period=12, extrapolate='fill_forward')
        return {
            'trend': decomp.trend,
            'seasonal': decomp.seasonal,
            'residual': decomp.resid,
        }

    def get_summary(self):
        """Get model summary statistics."""
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        return str(self.result.summary())


if __name__ == "__main__":
    # Test standalone
    from engine.ingestion import load_and_merge_data, validate_data
    from engine.feature_engineering import prepare_features

    # Load and prepare data
    df_raw = load_and_merge_data()
    validate_data(df_raw)
    features_dict = prepare_features(df_raw)

    # Use training set
    y_train = features_dict['y_train']
    X_train = features_dict['X_train']

    # Create exogenous variables (using subset of features: GDP growth, fuel change, COVID)
    # For now, we'll just use the training features directly
    df_full = features_dict['df_full']
    df_train = df_full.iloc[:len(y_train)].copy()
    exog_train = df_train[['gdp_growth', 'fuel_change', 'is_covid']].fillna(0).values

    # Fit ARIMA
    worker = ARIMAWorker()
    worker.fit(y_train, exog=exog_train)

    print("[OK] ARIMA model fitted successfully")
    print(f"Number of in-sample residuals: {len(worker.result.resid)}")

    # Test prediction
    exog_test = df_full.iloc[len(y_train):len(y_train)+12][['gdp_growth', 'fuel_change', 'is_covid']].fillna(0).values
    forecast, residuals = worker.predict(steps=12, exog_future=exog_test)

    print(f"Forecast shape: {forecast.shape}")
    print(f"Forecast (first 5): {forecast[:5]}")
    print(f"Residuals shape: {residuals.shape}")
    print(f"Residuals summary - mean: {residuals.mean():.4f}, std: {residuals.std():.4f}")
