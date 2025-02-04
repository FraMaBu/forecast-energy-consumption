# backtest.py
import numpy as np
from tensorflow.keras.models import load_model

from utils.data import download_data, scale_prices, create_sliding_windows
from utils.trading import backtest_strategy, plot_backtest
from utils.model import (
    PCAInitializer,
)  # Needed for custom_objects when loading the hybrid model


def main():
    # Data download and preprocessing
    prices, dates = download_data("BTC-USD", "2020-01-01", "2025-01-01")
    prices_scaled, scaler = scale_prices(prices)
    window_size = 60
    num_fourier = 5
    X_seq, X_fourier, y = create_sliding_windows(
        prices_scaled, window_size, num_fourier
    )

    # Train/test split (80% train, 20% test)
    train_size = int(len(X_seq) * 0.8)
    X_seq_test = X_seq[train_size:]
    X_fourier_test = X_fourier[train_size:]
    y_test = y[train_size:]

    # -------------------------------
    # Load Saved Models
    # -------------------------------
    hybrid_model = load_model(
        "models/hybrid_model.keras", custom_objects={"PCAInitializer": PCAInitializer}
    )
    simple_model = load_model("models/simple_model.keras")

    # -------------------------------
    # Generate Predictions on Test Data
    # -------------------------------
    y_pred_hybrid = hybrid_model.predict([X_seq_test, X_fourier_test])
    y_pred_simple = simple_model.predict(X_seq_test)

    # -------------------------------
    # Invert Scaling for Backtesting (Original Price Scale)
    # -------------------------------
    # For backtesting, we derive entry prices from the test windows and exit prices from y_test.
    actual_entry_prices = scaler.inverse_transform(
        X_seq_test[:, -1, 0].reshape(-1, 1)
    ).flatten()
    actual_exit_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    # Combine to form a price series (length n+1)
    actual_prices_backtest = np.concatenate(
        [actual_entry_prices, actual_exit_prices[-1:]]
    )

    y_pred_hybrid_orig = scaler.inverse_transform(y_pred_hybrid)
    y_pred_simple_orig = scaler.inverse_transform(y_pred_simple)
    y_pred_naive_orig = scaler.inverse_transform(
        X_seq_test[:, -1, 0].reshape(-1, 1)
    ).flatten()

    # -------------------------------
    # Run Backtesting for Each Forecasting Approach
    # -------------------------------
    portfolio_hybrid, _ = backtest_strategy(
        actual_prices_backtest,
        y_pred_hybrid_orig.flatten(),
        initial_capital=1000.0,
        threshold=0.0,
    )
    portfolio_simple, _ = backtest_strategy(
        actual_prices_backtest,
        y_pred_simple_orig.flatten(),
        initial_capital=1000.0,
        threshold=0.0,
    )
    portfolio_naive, _ = backtest_strategy(
        actual_prices_backtest,
        y_pred_naive_orig.flatten(),
        initial_capital=1000.0,
        threshold=0.0,
    )

    # Print final portfolio values
    print("Hybrid Model Backtest Final Portfolio Value:", portfolio_hybrid[-1])
    print("Simple Model Backtest Final Portfolio Value:", portfolio_simple[-1])
    print("Naive Baseline Backtest Final Portfolio Value:", portfolio_naive[-1])

    # Plot portfolio performance
    plot_backtest(portfolio_hybrid, title="Hybrid Model Backtest Portfolio Value")
    plot_backtest(portfolio_simple, title="Simple Model Backtest Portfolio Value")
    plot_backtest(portfolio_naive, title="Naive Baseline Backtest Portfolio Value")


if __name__ == "__main__":
    main()
