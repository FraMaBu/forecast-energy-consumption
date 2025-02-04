import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def download_data(ticker="BTC-USD", start="2020-01-01", end="2025-01-01"):
    """
    Download historical data for a given ticker.
    Returns the close prices and corresponding dates.
    """
    data = yf.download(ticker, start=start, end=end)
    prices = data["Close"].values
    dates = data.index
    return prices, dates


def scale_prices(prices):
    """
    Scales the prices between 0 and 1.
    Returns the scaled prices and the scaler (to invert scaling later if needed).
    """
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    return prices_scaled, scaler


def create_sliding_windows(prices_scaled, window_size=60, num_fourier=5):
    """
    Creates sliding windows from the scaled price series.
    Also computes Fourier features for each window (skipping the DC term).
    Returns:
      - X_seq: raw time series windows (shape: samples x window_size x 1)
      - X_fourier: Fourier features (shape: samples x num_fourier)
      - y: targets (the next time-step value)
    """
    X_seq, X_fourier, y = [], [], []
    for i in range(len(prices_scaled) - window_size):
        window = prices_scaled[i : i + window_size]
        target = prices_scaled[i + window_size]
        X_seq.append(window)

        fft_vals = np.fft.fft(window)
        fft_abs = np.abs(fft_vals)[1 : num_fourier + 1]  # skip the zero-frequency term
        X_fourier.append(fft_abs)

        y.append(target)

    X_seq = np.array(X_seq)[..., np.newaxis]  # shape: (samples, window_size, 1)
    X_fourier = np.array(X_fourier)
    y = np.array(y)

    return X_seq, X_fourier, y
