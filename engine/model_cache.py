"""
Model Cache Module
Handles saving/loading of trained model artifacts.
Cache invalidation based on CSV file modification times.
"""
import os
import hashlib
import json
import joblib
import config


class ModelCache:
    """Persistent model storage with cache invalidation."""

    @staticmethod
    def _get_data_hash(csv_paths):
        """
        Compute MD5 hash of CSV file modification times.
        Used to detect when data has changed.
        """
        h = hashlib.md5()
        for path in csv_paths:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                h.update(str(mtime).encode())
        return h.hexdigest()

    @staticmethod
    def save_arima(model):
        """Save ARIMA result."""
        joblib.dump(model, config.ARIMA_MODEL_PATH)
        print(f"[OK] ARIMA model saved to {config.ARIMA_MODEL_PATH}")

    @staticmethod
    def load_arima():
        """Load ARIMA result."""
        if os.path.exists(config.ARIMA_MODEL_PATH):
            return joblib.load(config.ARIMA_MODEL_PATH)
        return None

    @staticmethod
    def save_lstm(model, path=None):
        """
        Save LSTM model (sklearn MLPRegressor).
        Args:
            model: MLPRegressor model object
            path: optional path override
        """
        save_path = path or config.LSTM_MODEL_PATH
        joblib.dump(model, save_path)
        print(f"[OK] LSTM model saved to {save_path}")

    @staticmethod
    def load_lstm(path=None):
        """Load LSTM model."""
        load_path = path or config.LSTM_MODEL_PATH
        if os.path.exists(load_path):
            return joblib.load(load_path)
        return None

    @staticmethod
    def save_xgboost(model):
        """Save XGBoost model."""
        model.model.save_model(config.XGBOOST_MODEL_PATH)
        print(f"[OK] XGBoost model saved to {config.XGBOOST_MODEL_PATH}")

    @staticmethod
    def load_xgboost():
        """Load XGBoost model."""
        import xgboost as xgb
        if os.path.exists(config.XGBOOST_MODEL_PATH):
            model = xgb.XGBRegressor()
            model.load_model(config.XGBOOST_MODEL_PATH)
            return model
        return None

    @staticmethod
    def save_meta_learner(meta_learner):
        """Save MetaLearner (weights)."""
        weights = meta_learner.get_weights()
        joblib.dump(weights, config.META_LEARNER_PATH)
        print(f"[OK] MetaLearner saved to {config.META_LEARNER_PATH}")

    @staticmethod
    def load_meta_learner():
        """Load MetaLearner weights."""
        if os.path.exists(config.META_LEARNER_PATH):
            return joblib.load(config.META_LEARNER_PATH)
        return None

    @staticmethod
    def save_scaler(scaler):
        """Save features scaler."""
        joblib.dump(scaler, config.FEATURES_SCALER_PATH)

    @staticmethod
    def load_scaler():
        """Load features scaler."""
        if os.path.exists(config.FEATURES_SCALER_PATH):
            return joblib.load(config.FEATURES_SCALER_PATH)
        return None

    @staticmethod
    def save_data_hash(csv_paths):
        """Save hash of data files for cache invalidation."""
        h = ModelCache._get_data_hash(csv_paths)
        with open(config.DATA_HASH_PATH, 'w') as f:
            f.write(h)

    @staticmethod
    def check_cache_valid(csv_paths):
        """
        Check if cached models are still valid (data hasn't changed).
        """
        if not os.path.exists(config.DATA_HASH_PATH):
            return False

        try:
            with open(config.DATA_HASH_PATH, 'r') as f:
                old_hash = f.read().strip()
            new_hash = ModelCache._get_data_hash(csv_paths)
            return old_hash == new_hash
        except Exception:
            return False

    @staticmethod
    def clear_all():
        """Remove all cached models."""
        paths = [
            config.ARIMA_MODEL_PATH,
            config.LSTM_MODEL_PATH,
            config.XGBOOST_MODEL_PATH,
            config.META_LEARNER_PATH,
            config.FEATURES_SCALER_PATH,
            config.DATA_HASH_PATH
        ]
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
                print(f"[OK] Removed {path}")
