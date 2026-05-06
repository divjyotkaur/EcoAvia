"""
XGBoost Worker Module
Gradient boosting tree model for capturing non-linear interactions in features.
Used both as standalone model and as residual corrector in the hybrid ensemble.
"""
import numpy as np
import xgboost as xgb
import config


class XGBoostWorker:
    """XGBoost regressor for demand forecasting."""

    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=config.XGB_N_ESTIMATORS,
            learning_rate=config.XGB_LEARNING_RATE,
            max_depth=config.XGB_MAX_DEPTH,
            subsample=config.XGB_SUBSAMPLE,
            colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
            early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
            eval_metric='rmse',
            random_state=42,
            verbose=0
        )
        self.fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit XGBoost on training data.

        Args:
            X_train: (n_train, n_features)
            y_train: (n_train,)
            X_val: validation features (for early stopping)
            y_val: validation targets

        Returns:
            self
        """
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.fitted = True
        return self

    def predict(self, X):
        """
        Generate predictions.

        Args:
            X: (n_samples, n_features)

        Returns:
            predictions: (n_samples,)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self.model.predict(X)

    def get_feature_importance(self, feature_names=None):
        """
        Get feature importances (Gain).

        Args:
            feature_names: optional list of feature names

        Returns:
            dict with feature names and importance scores
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted.")

        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        return dict(zip(feature_names, importances))

    def get_shap_explainer(self):
        """
        Create SHAP TreeExplainer for feature attribution.

        Returns:
            shap.TreeExplainer object
        """
        import shap
        return shap.TreeExplainer(self.model)

    def get_summary(self):
        """Get model summary."""
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        return {
            'n_trees': self.model.n_estimators,
            'best_score': self.model.best_score if hasattr(self.model, 'best_score') else None,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None,
        }


if __name__ == "__main__":
    # Test standalone
    from engine.ingestion import load_and_merge_data, validate_data
    from engine.feature_engineering import prepare_features

    # Load and prepare data
    df_raw = load_and_merge_data()
    validate_data(df_raw)
    features_dict = prepare_features(df_raw)

    X_train = features_dict['X_train']
    y_train = features_dict['y_train']
    X_val = features_dict['X_val']
    y_val = features_dict['y_val']
    X_test = features_dict['X_test']
    y_test = features_dict['y_test']
    feature_cols = features_dict['feature_cols']

    # Fit XGBoost
    worker = XGBoostWorker()
    print("[XGBoost] Training...")
    worker.fit(X_train, y_train, X_val, y_val)

    # Predict
    print("[XGBoost] Predicting...")
    predictions = worker.predict(X_test)

    print(f"[OK] XGBoost predictions shape: {predictions.shape}")
    print(f"Predictions (first 5): {predictions[:5]}")

    # Get feature importance
    importance_dict = worker.get_feature_importance(feature_cols)
    print("\nTop 5 important features:")
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for feat, importance in sorted_importance[:5]:
        print(f"  {feat}: {importance:.4f}")
