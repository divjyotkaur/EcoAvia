"""
EcoAvia Configuration Module
Central repository for all paper constants and hyperparameters
"""
import os

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = "sqlite:///ecoavia.db"

# ==================== DATA SPLIT ====================
TRAIN_SPLIT = 0.60
VAL_SPLIT = 0.20
TEST_SPLIT = 0.20

# ==================== FORECAST HORIZON ====================
FORECAST_HORIZON_MONTHS = 24

# ==================== ARIMA CONFIGURATION ====================
ARIMA_ORDER = (1, 1, 1)
ARIMA_SEASONAL_ORDER = (1, 1, 1, 12)  # P, D, Q, s=12 (monthly seasonality)
COVID_START_MONTH = "2020-03"
COVID_END_MONTH = "2021-06"

# ==================== LSTM CONFIGURATION ====================
LSTM_LOOKBACK = 12  # 12-month window
LSTM_HIDDEN_UNITS = [64, 32]  # Two layers: 64 → 32
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 200
LSTM_BATCH_SIZE = 16
LSTM_LEARNING_RATE = 0.001
LSTM_EARLY_STOPPING_PATIENCE = 15

# ==================== XGBOOST CONFIGURATION ====================
XGB_N_ESTIMATORS = 1000
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 6
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_EARLY_STOPPING_ROUNDS = 50

# ==================== SUSTAINABILITY - CO2 & EMISSIONS ====================
ICAO_EMISSION_FACTOR_KG_PER_KG = 2.54  # kg CO2 per kg Jet-A (ICAO standard)
JET_FUEL_DENSITY_KG_PER_LITER = 0.8
AVG_STAGE_LENGTH_KM = 2500  # Average SFO stage length
FUEL_BURN_L_PER_100_KM_PER_PAX = 3.5  # Narrow-body estimate per passenger

# ==================== SUSTAINABILITY - INFRASTRUCTURE STRESS INDEX (ISI) ====================
SFO_TERMINAL_CAPACITY_MILLIONS = 57.0  # Million passengers per year
ISI_NORMAL_THRESHOLD = 0.70  # < 0.70: Normal (green)
ISI_WARNING_THRESHOLD = 0.85  # 0.70-0.85: Warning (yellow)
ISI_CRITICAL_THRESHOLD = 0.85  # > 0.85: Critical (red)

# ==================== SUSTAINABILITY - PCES (PASSENGER CARBON EFFICIENCY) ====================
BASELINE_CO2_PER_PAX_KG = 95.0  # IATA 2019 industry average
SAF_EMISSION_REDUCTION_FACTOR = 0.8  # SAF reduces CO2 by up to 80% vs Jet-A
DEFAULT_LOAD_FACTOR = 0.85  # Average aircraft load factor

# ==================== MODEL PERSISTENCE ====================
ARIMA_MODEL_PATH = os.path.join(MODELS_DIR, "arima_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.pt")
XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")
META_LEARNER_PATH = os.path.join(MODELS_DIR, "meta_learner.pkl")
FEATURES_SCALER_PATH = os.path.join(MODELS_DIR, "features_scaler.pkl")

# ==================== DATA HASHING FOR CACHE INVALIDATION ====================
DATA_HASH_PATH = os.path.join(MODELS_DIR, "data_hash.txt")

# ==================== FEATURE ENGINEERING ====================
LAG_FEATURES = [1, 2, 3, 6, 12]  # Lag 1, 2, 3, 6, 12 months
ROLLING_WINDOW_MEAN = 3  # 3-month rolling mean
ROLLING_WINDOW_STD = 6  # 6-month rolling std

# ==================== PRESCRIPTIVE ENGINE ====================
FUEL_HEDGING_TRIGGER_PRICE = 2.80  # $/gal
FUEL_HEDGING_DEMAND_THRESHOLD = 0.04  # 4% YoY growth
FUEL_HEDGING_COVER_RATIO = 0.65  # Hedge 65% of forward fuel needs

# ==================== CSV COLUMN NAMES ====================
# Expected columns in ingested data
TRAFFIC_CSV_COLUMNS = ["Activity Period Start Date", "Passenger Count"]
GDP_CSV_COLUMNS = ["observation_date", "GDP"]
FUEL_CSV_COLUMNS = ["observation_date", "DJFUELUSGULF"]
