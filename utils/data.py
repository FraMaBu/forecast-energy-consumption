import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_power_data(filepath="data/household_power_consumption.txt"):
    """
    Loads the Individual Household Electric Power Consumption dataset from a local file.

    The dataset is semicolon-delimited with missing values coded as '?'.
    This function parses the Date and Time columns into a single Datetime index,
    drops rows with missing values, and then resamples the data to hourly means.

    Returns:
      - data: numpy array of hourly average Global_active_power values.
      - dates: DatetimeIndex corresponding to the hourly timestamps.
    """
    # Read the raw data from the local file.
    df = pd.read_csv(
        filepath,
        sep=";",
        na_values="?",
        low_memory=False,
    )

    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", dayfirst=True
    )

    # Optionally drop the original 'Date' and 'Time' columns.
    df.drop(columns=["Date", "Time"], inplace=True)

    # Drop rows with missing values and set index.
    df = df.dropna()
    df.set_index("Datetime", inplace=True)

    # Resample to hourly frequency by taking the mean and drop hours with NaNs.
    hourly_df = df.resample("h").mean().dropna()
    data = hourly_df["Global_active_power"].values
    dates = hourly_df.index
    return data, dates


def scale_data(data):
    """
    Scales the data between 0 and 1.

    Returns:
      - data_scaled: the scaled 1D numpy array.
      - scaler: the fitted MinMaxScaler.
    """
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    return data_scaled, scaler


def compute_fourier_features(
    timestamp, daily_harmonics=3, weekly_harmonics=2, annual_harmonics=1
):
    """
    Computes deterministic Fourier features for a given timestamp based on
    daily, weekly, and annual cycles.

    Parameters:
        timestamp: A pandas Timestamp (or datetime object).
        daily_harmonics: Number of harmonics for the daily cycle (period = 24 hours).
        weekly_harmonics: Number of harmonics for the weekly cycle (period = 168 hours).
        annual_harmonics: Number of harmonics for the annual cycle (period = 8760 hours).

    Returns:
        A numpy array of Fourier features
    """
    features = []

    # Daily cycle: period = 24 hours.
    t_daily = timestamp.hour
    for k in range(1, daily_harmonics + 1):
        features.append(np.sin(2 * np.pi * k * t_daily / 24))
        features.append(np.cos(2 * np.pi * k * t_daily / 24))

    # Weekly cycle: period = 168 hours (7 days).
    t_weekly = timestamp.dayofweek * 24 + timestamp.hour
    for k in range(1, weekly_harmonics + 1):
        features.append(np.sin(2 * np.pi * k * t_weekly / 168))
        features.append(np.cos(2 * np.pi * k * t_weekly / 168))

    # Annual cycle: period = 8760 hours (365 days).
    t_annual = (timestamp.timetuple().tm_yday - 1) * 24 + timestamp.hour
    for k in range(1, annual_harmonics + 1):
        features.append(np.sin(2 * np.pi * k * t_annual / 8760))
        features.append(np.cos(2 * np.pi * k * t_annual / 8760))

    return np.array(features)


def create_sliding_windows(
    data_scaled,
    dates,
    window_size=168,
    daily_harmonics=3,
    weekly_harmonics=2,
    annual_harmonics=1,
):
    """
    Creates sliding windows from the scaled time series and deterministically computes Fourier features
    for the target timestamp of each window.

    For each sliding window, the corresponding Fourier features are computed for the forecast target time
    (i.e. the time immediately following the window) using predetermined daily, weekly, and annual cycles.

    Returns:
      - X_seq: raw time series windows (samples x window_size x 1)
      - X_fourier: Deterministic Fourier features (samples x num_features)
      - y: targets (the value following each window)
    """
    X_seq, X_fourier, y = [], [], []
    for i in range(len(data_scaled) - window_size):
        window = data_scaled[i : i + window_size]
        target = data_scaled[i + window_size]
        # Skip window if any NaN is present (for safety)
        if np.isnan(window).any() or np.isnan(target):
            continue
        X_seq.append(window)

        # Compute deterministic Fourier features using the target timestamp.
        ft_features = compute_fourier_features(
            dates[i + window_size], daily_harmonics, weekly_harmonics, annual_harmonics
        )
        X_fourier.append(ft_features)

        y.append(target)

    X_seq = np.array(X_seq)[..., np.newaxis]  # Shape: (samples, window_size, 1)
    X_fourier = np.array(X_fourier)
    y = np.array(y)

    return X_seq, X_fourier, y
