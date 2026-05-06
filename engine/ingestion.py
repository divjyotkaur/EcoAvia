"""
ETL & Data Ingestion Module
Loads, cleans, and merges data from three CSV sources:
- Passenger traffic (monthly)
- GDP (quarterly, forward-filled to monthly)
- Jet fuel prices (weekly, resampled to monthly mean)
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config


def load_traffic_data():
    """Load and clean passenger traffic data."""
    csv_path = os.path.join(config.DATA_DIR, "Air_Traffic_Passenger_Statistics.csv")

    try:
        df = pd.read_csv(csv_path, usecols=config.TRAFFIC_CSV_COLUMNS)
        # Clean: remove commas from Passenger Count
        df[config.TRAFFIC_CSV_COLUMNS[1]] = (
            df[config.TRAFFIC_CSV_COLUMNS[1]].astype(str)
            .str.replace(',', '')
            .astype(float, errors='coerce')
        )
        df = df.dropna()
        df[config.TRAFFIC_CSV_COLUMNS[0]] = pd.to_datetime(
            df[config.TRAFFIC_CSV_COLUMNS[0]]
        )
        # Aggregate to monthly
        df.set_index(config.TRAFFIC_CSV_COLUMNS[0], inplace=True)
        monthly = df.resample('MS')[config.TRAFFIC_CSV_COLUMNS[1]].sum()
        return monthly / 1_000_000  # Convert to millions
    except FileNotFoundError:
        print(f"[WARNING] Traffic CSV not found at {csv_path}")
        return generate_synthetic_traffic_data()


def load_gdp_data():
    """Load and clean GDP data (quarterly → forward-fill to monthly)."""
    csv_path = os.path.join(config.DATA_DIR, "GDP_ecoavia.csv")

    try:
        df = pd.read_csv(csv_path, usecols=config.GDP_CSV_COLUMNS)
        df[config.GDP_CSV_COLUMNS[0]] = pd.to_datetime(
            df[config.GDP_CSV_COLUMNS[0]]
        )
        df[config.GDP_CSV_COLUMNS[1]] = df[config.GDP_CSV_COLUMNS[1]].astype(float)
        df.set_index(config.GDP_CSV_COLUMNS[0], inplace=True)
        # Resample to monthly, forward-fill
        monthly = df.resample('MS')[config.GDP_CSV_COLUMNS[1]].last().fillna(method='ffill')
        return monthly
    except FileNotFoundError:
        print(f"[WARNING] GDP CSV not found at {csv_path}")
        return generate_synthetic_gdp_data()


def load_fuel_data():
    """Load and clean jet fuel price data (weekly → monthly mean)."""
    csv_path = os.path.join(config.DATA_DIR, "Jet_Fuel_ecoavia.csv")

    try:
        df = pd.read_csv(csv_path, usecols=config.FUEL_CSV_COLUMNS)
        df[config.FUEL_CSV_COLUMNS[0]] = pd.to_datetime(
            df[config.FUEL_CSV_COLUMNS[0]]
        )
        df[config.FUEL_CSV_COLUMNS[1]] = df[config.FUEL_CSV_COLUMNS[1]].astype(float)
        df.set_index(config.FUEL_CSV_COLUMNS[0], inplace=True)
        df = df.dropna()
        # Resample weekly data to monthly mean
        monthly = df.resample('MS')[config.FUEL_CSV_COLUMNS[1]].mean()
        return monthly
    except FileNotFoundError:
        print(f"[WARNING] Fuel CSV not found at {csv_path}")
        return generate_synthetic_fuel_data()


def generate_synthetic_traffic_data(n_months=120):
    """Generate synthetic SFO-like passenger traffic data."""
    dates = pd.date_range(start='2016-07-01', periods=n_months, freq='MS')
    # Trend: 3.5M base + 0.2% monthly growth + seasonality
    trend = 3.5 + np.arange(n_months) * 0.002
    seasonality = 0.3 * np.sin(np.arange(n_months) * 2 * np.pi / 12)
    noise = np.random.normal(0, 0.1, n_months)
    passengers = trend + seasonality + noise
    passengers = np.clip(passengers, 0.5, 5.0)  # Realistic bounds
    return pd.Series(passengers, index=dates)


def generate_synthetic_gdp_data(n_months=120):
    """Generate synthetic US GDP data (in billions)."""
    dates = pd.date_range(start='2016-07-01', periods=n_months, freq='MS')
    # GDP trend: 25000B base + ~0.5% quarterly growth
    gdp_values = 25000 + np.arange(n_months) * 10
    noise = np.random.normal(0, 50, n_months)
    gdp = gdp_values + noise
    return pd.Series(gdp, index=dates)


def generate_synthetic_fuel_data(n_months=120):
    """Generate synthetic jet fuel price data (in $/gal)."""
    dates = pd.date_range(start='2016-07-01', periods=n_months, freq='MS')
    # Fuel price: 2.0-3.5 $/gal range with volatility
    base = 2.5
    trend = 0.0015 * np.arange(n_months)  # Slight uptrend
    seasonality = 0.3 * np.sin(np.arange(n_months) * 2 * np.pi / 12)
    noise = np.random.normal(0, 0.15, n_months)
    prices = base + trend + seasonality + noise
    prices = np.clip(prices, 1.5, 4.0)
    return pd.Series(prices, index=dates)


def add_covid_dummy(df, date_index):
    """Add COVID-19 shock indicator variable."""
    covid_start = pd.Timestamp(config.COVID_START_MONTH)
    covid_end = pd.Timestamp(config.COVID_END_MONTH)
    is_covid = ((date_index >= covid_start) & (date_index <= covid_end)).astype(int)
    df['is_covid'] = is_covid
    return df


def load_and_merge_data():
    """
    Load all three data sources and merge on monthly index.
    Returns: DataFrame with columns [date, passengers_M, gdp, fuel_price, is_covid]
    """
    # Load individual datasets
    traffic = load_traffic_data()
    gdp = load_gdp_data()
    fuel = load_fuel_data()

    # Create unified monthly index (from earliest to latest)
    all_dates = traffic.index.union(gdp.index).union(fuel.index)
    all_dates = all_dates.sort_values()

    # Merge on monthly index
    df = pd.DataFrame(index=all_dates)
    df['passengers_M'] = traffic.reindex(all_dates)
    df['gdp'] = gdp.reindex(all_dates).ffill()  # Forward-fill quarterly GDP
    df['fuel_price'] = fuel.reindex(all_dates).ffill()  # Forward-fill fuel

    # Drop rows with NaN values
    df = df.dropna()

    # Add COVID dummy
    df = add_covid_dummy(df, df.index)

    # Reset index
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)

    # Ensure chronological order
    df = df.sort_values('date').reset_index(drop=True)

    return df


def validate_data(df):
    """Perform basic data quality checks."""
    assert len(df) > 0, "Dataset is empty"
    assert not df[['passengers_M', 'gdp', 'fuel_price']].isna().any().any(), \
        "Dataset contains NaN values"
    assert (df['passengers_M'] > 0).all(), "Negative passenger counts found"
    assert (df['fuel_price'] > 0).all(), "Negative fuel prices found"
    assert (df['date'] == df['date'].sort_values()).all(), "Date column not sorted"
    print(f"[OK] Data validation passed. {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    # Test standalone
    df = load_and_merge_data()
    validate_data(df)
    print(df.head(10))
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
