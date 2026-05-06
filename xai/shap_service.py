"""SHAP Explainability Service for XGBoost Model."""
import shap
import pandas as pd


class SHAPService:
    """Generate SHAP explanations for model predictions."""

    def __init__(self, xgboost_model):
        """
        Args:
            xgboost_model: fitted XGBoost model
        """
        self.model = xgboost_model
        self.explainer = shap.TreeExplainer(xgboost_model)

    def explain(self, X, feature_names=None, top_n=5):
        """
        Compute SHAP values and return top N features.

        Args:
            X: input data (n_samples, n_features)
            feature_names: optional feature names
            top_n: number of top features to return

        Returns:
            list of dicts with feature name and importance
        """
        try:
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X)

            # If shap_values is list (multiclass), take first element
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Mean absolute SHAP value per feature
            mean_abs_shap = pd.Series(
                abs(shap_values).mean(axis=0),
                index=feature_names if feature_names else [f'feature_{i}' for i in range(shap_values.shape[1])]
            ).sort_values(ascending=False)

            # Return top N
            result = [
                {'feature': name, 'importance': float(importance)}
                for name, importance in mean_abs_shap.head(top_n).items()
            ]
            return result
        except Exception as e:
            print(f"[WARNING] SHAP explanation failed: {e}")
            return []
