"""
EcoAvia Flask Application
Routes requests to the real forecasting engine
"""
from flask import Flask, render_template, jsonify
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.forecaster import Forecaster
from prescriptive.optimizer import PrescriptiveOptimizer

app = Flask(__name__)
forecaster = Forecaster()

# Cache for forecast results
_cached_forecast = None


@app.route('/')
def index():
    """Serve main dashboard."""
    return render_template('index.html')


@app.route('/api/data')
def get_data():
    """
    Main API endpoint returning all forecast data.
    Combines real ML predictions with sustainability metrics.
    """
    try:
        global _cached_forecast

        # Run forecasting pipeline (with caching)
        if _cached_forecast is None:
            print("[API] Running forecasting pipeline...")
            _cached_forecast = forecaster.run()

        forecast_result = _cached_forecast

        # Generate recommendations
        sustainability = forecast_result.get('sustainability', {})
        recommendations = PrescriptiveOptimizer.generate_recommendations(
            forecast_result, sustainability
        )

        # Add recommendations to response
        forecast_result['recommendations'] = recommendations

        return jsonify(forecast_result)

    except Exception as e:
        print(f"[ERROR] API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/forecast')
def get_forecast():
    """
    Get 24-month forward forecast with confidence intervals.
    """
    try:
        global _cached_forecast
        if _cached_forecast is None:
            _cached_forecast = forecaster.run()

        result = _cached_forecast

        return jsonify({
            'forecast_months': len(result.get('predicted_data', [])),
            'forecast_values': result.get('predicted_data', []),
            'confidence_upper': result.get('confidence_upper', []),
            'confidence_lower': result.get('confidence_lower', []),
            'dates': [d for d in result.get('historical_dates', [])][-24:] if result.get('historical_dates') else []
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/sustainability')
def get_sustainability():
    """Get sustainability metrics (CO2, ISI, PCES)."""
    try:
        global _cached_forecast
        if _cached_forecast is None:
            _cached_forecast = forecaster.run()

        sustainability = _cached_forecast.get('sustainability', {})
        return jsonify(sustainability)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/shap')
def get_shap():
    """Get SHAP feature importances."""
    try:
        global _cached_forecast
        if _cached_forecast is None:
            _cached_forecast = forecaster.run()

        shap_features = _cached_forecast.get('shap_features', [])
        return jsonify({'features': shap_features})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/model/status')
def model_status():
    """Get model training status and metrics."""
    try:
        global _cached_forecast
        if _cached_forecast is None:
            _cached_forecast = forecaster.run()

        metrics = _cached_forecast.get('model_metrics', {})
        weights = _cached_forecast.get('meta_weights', {})

        return jsonify({
            'status': 'trained',
            'hybrid_metrics': metrics.get('hybrid', {}),
            'meta_weights': weights,
            'models_trained': ['ARIMA', 'LSTM', 'XGBoost', 'MetaLearner']
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/simulate', methods=['POST'])
def simulate_scenario():
    """
    What-if scenario simulator.
    POST body: {fuel_multiplier: float, gdp_delta: float, saf_pct: float}
    """
    from flask import request

    try:
        data = request.get_json()

        # For now, return static recommendations
        # In production, re-run forecaster with modified parameters
        recommendations = [
            {
                'priority': 'High',
                'category': 'Fuel',
                'action': f'Fuel scenario: {data.get("fuel_multiplier", 1.0)}x impact on hedging strategy',
                'impact': 'Scenario-based analysis'
            }
        ]

        return jsonify({'status': 'simulated', 'recommendations': recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Server error"}), 500


if __name__ == '__main__':
    print("[INFO] Starting EcoAvia Flask server...")
    print("[INFO] Ensure data CSVs are in ./data/ folder")
    app.run(debug=True, port=5000, threaded=True)
