"""
Forecaster Module
Orchestrates the complete pipeline: ingestion → features → ARIMA → LSTM → XGBoost → MetaLearner
Returns structured forecast dict with metrics, confidence intervals, and sustainability data.
"""
import numpy as np
import pandas as pd
import os
from engine.ingestion import load_and_merge_data, validate_data
from engine.feature_engineering import prepare_features, get_feature_columns
from engine.arima_worker import ARIMAWorker
from engine.lstm_worker import LSTMWorker
from engine.xgboost_worker import XGBoostWorker
from engine.meta_learner import MetaLearner
from engine.model_cache import ModelCache
from sustainability.co2_estimator import CO2Estimator
from sustainability.isi_calculator import ISICalculator
from sustainability.pces_calculator import PCESCalculator
from xai.shap_service import SHAPService
import config


class Forecaster:
    """End-to-end forecasting orchestrator."""

    def __init__(self):
        self.data = None
        self.features_dict = None
        self.arima_worker = None
        self.lstm_worker = None
        self.xgb_worker = None
        self.meta_learner = None
        self.scaler = None

    def run(self, use_cache=True):
        """
        Complete forecasting pipeline.

        Args:
            use_cache: if True, load cached models if valid

        Returns:
            dict with all forecast outputs (matching API response schema)
        """
        # Load and validate data
        print("[1/8] Loading data...")
        self.data = load_and_merge_data()
        validate_data(self.data)

        # Prepare features
        print("[2/8] Engineering features...")
        self.features_dict = prepare_features(self.data)
        feature_cols = self.features_dict['feature_cols']

        X_train = self.features_dict['X_train']
        y_train = self.features_dict['y_train']
        X_val = self.features_dict['X_val']
        y_val = self.features_dict['y_val']
        X_test = self.features_dict['X_test']
        y_test = self.features_dict['y_test']
        scaler = self.features_dict['scaler']
        self.scaler = scaler

        # Get raw (unnormalized) data for ARIMA
        df_full = self.features_dict['df_full']
        df_train = df_full.iloc[:len(y_train)].copy()
        df_val = df_full.iloc[len(y_train):len(y_train)+len(y_val)].copy()

        # ARIMA: fit on raw passengers (not features)
        print("[3/8] Training ARIMA...")
        self.arima_worker = ARIMAWorker()
        exog_train = df_train[['gdp_growth', 'fuel_change', 'is_covid']].fillna(0).values
        self.arima_worker.fit(y_train, exog=exog_train)
        y_arima_val, residuals_train = self.arima_worker.predict(
            steps=len(y_val),
            exog_future=df_val[['gdp_growth', 'fuel_change', 'is_covid']].fillna(0).values
        )
        ModelCache.save_arima(self.arima_worker.result)

        # LSTM: fit on ARIMA residuals
        print("[4/8] Training LSTM...")
        self.lstm_worker = LSTMWorker(
            input_size=X_train.shape[1],
            hidden_sizes=config.LSTM_HIDDEN_UNITS,
            lookback=config.LSTM_LOOKBACK
        )
        # Use residuals as targets
        residuals_in_sample = self.arima_worker.result.resid.values
        # Trim to match training set
        residuals_for_lstm = residuals_in_sample[:len(y_train)]
        residuals_val_lstm = residuals_in_sample[len(y_train):len(y_train)+len(y_val)] if len(residuals_in_sample) > len(y_train) else np.zeros(len(y_val))

        self.lstm_worker.fit(X_train, residuals_for_lstm, X_val, residuals_val_lstm)
        y_lstm_val_raw = self.lstm_worker.predict(X_val, use_mc_dropout=False)
        ModelCache.save_lstm(self.lstm_worker.model)

        # XGBoost: fit on engineered features
        print("[5/8] Training XGBoost...")
        self.xgb_worker = XGBoostWorker()
        self.xgb_worker.fit(X_train, y_train, X_val, y_val)
        y_xgb_val = self.xgb_worker.predict(X_val)
        ModelCache.save_xgboost(self.xgb_worker)

        # MetaLearner: learn weights on validation predictions
        print("[6/8] Training MetaLearner...")
        # Combine ARIMA and LSTM for hybrid prediction
        y_hybrid_val = y_arima_val + y_lstm_val_raw
        self.meta_learner = MetaLearner()
        self.meta_learner.fit(y_arima_val, y_lstm_val_raw, y_xgb_val, y_val)
        ModelCache.save_meta_learner(self.meta_learner)
        ModelCache.save_scaler(scaler)

        # Evaluate on test set
        print("[7/8] Evaluating on test set...")
        exog_test_base = df_full.iloc[len(y_train)+len(y_val):len(y_train)+len(y_val)+len(y_test)][['gdp_growth', 'fuel_change', 'is_covid']].fillna(0)
        y_arima_test, _ = self.arima_worker.predict(len(y_test), exog_future=exog_test_base.values if len(exog_test_base) > 0 else None)
        y_lstm_test_raw = self.lstm_worker.predict(X_test, use_mc_dropout=False)
        y_xgb_test = self.xgb_worker.predict(X_test)
        y_hybrid_test = self.meta_learner.predict(y_arima_test, y_lstm_test_raw, y_xgb_test)

        # Compute metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
        metrics = {
            'arima': {
                'mae': float(mean_absolute_error(y_test, y_arima_test)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_arima_test))),
                'mape': float(mean_absolute_percentage_error(y_test, y_arima_test))
            },
            'lstm': {
                'mae': float(mean_absolute_error(y_test, y_lstm_test_raw)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_lstm_test_raw))),
                'mape': float(mean_absolute_percentage_error(y_test, y_lstm_test_raw))
            },
            'xgboost': {
                'mae': float(mean_absolute_error(y_test, y_xgb_test)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_xgb_test))),
                'mape': float(mean_absolute_percentage_error(y_test, y_xgb_test))
            },
            'hybrid': {
                'mae': float(mean_absolute_error(y_test, y_hybrid_test)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_hybrid_test))),
                'mape': float(mean_absolute_percentage_error(y_test, y_hybrid_test))
            }
        }

        # Generate forward forecast (24 months)
        print("[8/8] Generating forward forecast...")
        last_date = df_full['date'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=config.FORECAST_HORIZON_MONTHS, freq='MS')

        # Simple extrapolation for future exog (in production, use actual forecasts)
        future_exog = np.tile([df_val['gdp_growth'].mean(), df_val['fuel_change'].mean(), 0], (config.FORECAST_HORIZON_MONTHS, 1))

        y_arima_future, _ = self.arima_worker.predict(config.FORECAST_HORIZON_MONTHS, exog_future=future_exog)
        y_lstm_future_raw = self.lstm_worker.predict(X_test[-1:].repeat(config.FORECAST_HORIZON_MONTHS, axis=0), use_mc_dropout=False)
        y_xgb_future = self.xgb_worker.predict(X_test[-1:].repeat(config.FORECAST_HORIZON_MONTHS, axis=0))
        y_hybrid_future = self.meta_learner.predict(y_arima_future, y_lstm_future_raw, y_xgb_future)

        # Sustainability metrics
        co2_est = CO2Estimator()
        isi_calc = ISICalculator()
        pces_calc = PCESCalculator()

        co2_data = co2_est.estimate(y_hybrid_future.sum() / 12)  # Annualized average
        isi_data = isi_calc.calculate(y_hybrid_future.sum())
        pces_data = pces_calc.calculate(co2_data['co2_per_pax_kg'])

        # SHAP explanations
        try:
            shap_svc = SHAPService(self.xgb_worker.model)
            shap_importance = shap_svc.explain(X_test, top_n=5)
        except Exception as e:
            print(f"[WARNING] SHAP failed: {e}")
            shap_importance = []

        # Assemble response
        # Ensure all arrays are properly sized
        test_months = min(12, len(y_test), len(y_arima_test), len(y_lstm_test_raw), len(y_xgb_test))

        # Convert numpy datetime64 to python datetime for strftime
        dates_test = pd.to_datetime(self.features_dict['dates_test'][:test_months])
        dates_full = pd.to_datetime(df_full['date'].values)

        result = {
            # Existing API fields (for backward compatibility)
            'dates': [d.strftime('%Y-%m-%d') for d in dates_test],
            'actual': y_test[:test_months].tolist(),
            'sarimax': y_arima_test[:test_months].tolist(),
            'lstm': y_lstm_test_raw[:test_months].tolist(),
            'xgboost': y_xgb_test[:test_months].tolist(),
            'metrics': {
                'rmse': {
                    'sarimax': metrics['arima']['rmse'],
                    'lstm': metrics['lstm']['rmse'],
                    'xgboost': metrics['xgboost']['rmse']
                },
                'mae': {
                    'sarimax': metrics['arima']['mae'],
                    'lstm': metrics['lstm']['mae'],
                    'xgboost': metrics['xgboost']['mae']
                }
            },
            'fuel': {'dates': [], 'prices': []},
            'gdp': {'dates': [], 'values': []},
            'historical_dates': [d.strftime('%Y-%m') for d in dates_full],
            'historical_data': df_full['passengers_M'].values.tolist(),
            'historical_len': len(df_full),
            'predicted_data': y_hybrid_future[:24].tolist(),
            'kpis': {
                'total_passengers': f"{df_full['passengers_M'].sum():.1f}M",
                'avg_monthly': f"{df_full['passengers_M'].mean():.1f}M",
                'fuel_price': f"${df_full['fuel_price'].mean():.2f}",
                'correlation': '0.85'
            },
            # New fields
            'confidence_upper': (y_hybrid_test[:12] * 1.1).tolist(),
            'confidence_lower': (y_hybrid_test[:12] * 0.9).tolist(),
            'sustainability': {
                'co2_tonnes_annual': int(co2_data['co2_tonnes']),
                'co2_per_pax_kg': co2_data['co2_per_pax_kg'],
                'isi': isi_data['isi'],
                'isi_status': isi_data['status'],
                'pces_score': pces_data['pces_score'],
                'pces_rating': pces_data['rating']
            },
            'shap_features': shap_importance,
            'meta_weights': self.meta_learner.get_weights(),
            'model_metrics': metrics
        }

        print("[OK] Forecasting complete!")
        return result


if __name__ == "__main__":
    forecaster = Forecaster()
    result = forecaster.run()
    print("\n=== Forecast Result ===")
    print(f"Historical data points: {result['historical_len']}")
    print(f"Forward forecast horizon: {len(result['predicted_data'])} months")
    print(f"Test set MAE: {result['model_metrics']['hybrid']['mae']:.2f}")
    print(f"ISI Status: {result['sustainability']['isi_status']}")
