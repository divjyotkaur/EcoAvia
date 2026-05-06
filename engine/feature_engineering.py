"""
Feature Engineering Module
Creates lag features, rolling statistics, cyclical encoding, and external variables.
Handles 60/20/20 train/validation/test split with no lookahead bias.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import config


def create_lag_features(df, col, lags):
    """Create lag features for a column."""
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df


def create_rolling_features(df, col, windows):
    """Create rolling mean and std features."""
    for window in windows:
        df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
        df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
    return df


def create_cyclical_features(df):
    """Create cyclical encoding for month-of-year seasonality."""
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


def create_external_features(df):
    """Create external economic variables: GDP growth, fuel change, etc."""
    # Year-over-year GDP growth (4 quarters back)
    df['gdp_growth'] = df['gdp'].pct_change(periods=4)
    # Month-over-month fuel price change
    df['fuel_change'] = df['fuel_price'].pct_change(periods=1)
    return df


def engineer_features(df):
    """
    Complete feature engineering pipeline.
    Input: DataFrame with columns [date, passengers_M, gdp, fuel_price, is_covid]
    Output: DataFrame with engineered features, NaN rows dropped
    """
    df = df.copy()

    # Create lag features on passengers
    df = create_lag_features(df, 'passengers_M', config.LAG_FEATURES)

    # Create rolling statistics
    df = create_rolling_features(df, 'passengers_M',
                                  [config.ROLLING_WINDOW_MEAN, config.ROLLING_WINDOW_STD])

    # Create cyclical month encoding
    df = create_cyclical_features(df)

    # Create external variables
    df = create_external_features(df)

    # Drop rows with NaN (from lag features and rolling windows)
    # Minimum NaN rows = max(LAG_FEATURES) = 12 rows
    df = df.dropna()

    return df


def normalize_features(X_train, X_val, X_test):
    """
    Fit MinMaxScaler on training set, apply to all splits.
    Returns: normalized arrays + scaler (for persistence)
    """
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_val_norm, X_test_norm, scaler


def split_train_val_test(df, feature_cols):
    """
    Chronological train/val/test split: 60/20/20
    No lookahead bias (no random shuffling).

    Args:
        df: engineered DataFrame
        feature_cols: list of feature column names

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test (as arrays)
        train_idx, val_idx, test_idx (row indices)
    """
    n = len(df)
    train_size = int(n * config.TRAIN_SPLIT)
    val_size = int(n * config.VAL_SPLIT)

    train_idx = np.arange(0, train_size)
    val_idx = np.arange(train_size, train_size + val_size)
    test_idx = np.arange(train_size + val_size, n)

    X_train = df.iloc[train_idx][feature_cols].values
    y_train = df.iloc[train_idx]['passengers_M'].values

    X_val = df.iloc[val_idx][feature_cols].values
    y_val = df.iloc[val_idx]['passengers_M'].values

    X_test = df.iloc[test_idx][feature_cols].values
    y_test = df.iloc[test_idx]['passengers_M'].values

    return X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx


def get_feature_columns(df):
    """
    Return list of all engineered feature column names (exclude date/passengers_M/gdp/fuel_price).
    """
    exclude = {'date', 'passengers_M', 'gdp', 'fuel_price', 'month', 'is_covid'}
    feature_cols = [col for col in df.columns if col not in exclude]
    return feature_cols


def prepare_features(df):
    """
    Complete feature preparation pipeline: engineer → split → normalize

    Args:
        df: raw DataFrame from ingestion

    Returns:
        dict with:
        - 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test' (normalized arrays)
        - 'scaler' (MinMaxScaler for inverse transform)
        - 'feature_cols' (list of feature names)
        - 'df_full' (engineered DataFrame with all rows, for reference)
        - 'dates_train', 'dates_val', 'dates_test' (date objects for each split)
    """
    # Engineer features
    df_eng = engineer_features(df)

    # Get feature column names
    feature_cols = get_feature_columns(df_eng)

    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx = \
        split_train_val_test(df_eng, feature_cols)

    # Normalize
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(X_train, X_val, X_test)

    # Extract dates for each split
    dates_train = df_eng.iloc[train_idx]['date'].values
    dates_val = df_eng.iloc[val_idx]['date'].values
    dates_test = df_eng.iloc[test_idx]['date'].values

    return {
        'X_train': X_train_norm,
        'y_train': y_train,
        'X_val': X_val_norm,
        'y_val': y_val,
        'X_test': X_test_norm,
        'y_test': y_test,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'df_full': df_eng,
        'dates_train': dates_train,
        'dates_val': dates_val,
        'dates_test': dates_test,
    }


if __name__ == "__main__":
    # Test standalone
    from engine.ingestion import load_and_merge_data, validate_data

    df_raw = load_and_merge_data()
    validate_data(df_raw)

    features_dict = prepare_features(df_raw)

    print(f"X_train shape: {features_dict['X_train'].shape}")
    print(f"X_val shape: {features_dict['X_val'].shape}")
    print(f"X_test shape: {features_dict['X_test'].shape}")
    print(f"Feature columns: {features_dict['feature_cols']}")
    print(f"Total engineered rows: {len(features_dict['df_full'])}")
