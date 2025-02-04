# main.py
import numpy as np
from sklearn.metrics import mean_squared_error

from utils.data import download_data, scale_prices, create_sliding_windows
from utils.model import (
    build_preliminary_model,
    compute_pca_weights,
    build_hybrid_model,
    build_simple_model,
)
from utils.train import train_model, evaluate_model, plot_predictions, plot_comparison


def main():
    # Data download and preprocessing
    prices, dates = download_data("BTC-USD", "2018-01-01", "2024-01-01")
    prices_scaled, scaler = scale_prices(prices)
    window_size = 60
    num_fourier = 5
    X_seq, X_fourier, y = create_sliding_windows(
        prices_scaled, window_size, num_fourier
    )

    # Simple non-shuffled train/test split (80% train, 20% test)
    train_size = int(len(X_seq) * 0.8)
    X_seq_train, X_seq_test = X_seq[:train_size], X_seq[train_size:]
    X_fourier_train, X_fourier_test = X_fourier[:train_size], X_fourier[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Preliminary Model for PCA
    print("Building and training preliminary model...")
    prelim_model = build_preliminary_model(window_size)
    train_model(
        prelim_model,
        X_seq_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
    )

    # Compute PCA weights from the hidden states.
    pca_weights = compute_pca_weights(prelim_model, X_seq_train, n_components=20)

    # Hybrid Model
    print("Building and training hybrid model...")
    hybrid_model = build_hybrid_model(
        window_size, num_fourier, pca_weights, lstm_units=50, n_pca_components=20
    )
    train_model(
        hybrid_model,
        [X_seq_train, X_fourier_train],
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
    )
    hybrid_loss = evaluate_model(hybrid_model, [X_seq_test, X_fourier_test], y_test)
    print("Hybrid Model Test Loss:", hybrid_loss)
    y_pred_hybrid = hybrid_model.predict([X_seq_test, X_fourier_test])

    # Benchmarks
    # 1) Simple LSTM Model
    print("Building and training simple LSTM model...")
    simple_model = build_simple_model(window_size)
    train_model(
        simple_model,
        X_seq_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
    )
    simple_loss = evaluate_model(simple_model, X_seq_test, y_test)
    print("Simple Model Test Loss:", simple_loss)
    y_pred_simple = simple_model.predict(X_seq_test)

    # 2) Naive baseline: use the last observed value from each input window.
    y_pred_naive = X_seq_test[:, -1, 0]

    # Compute MSE for all models.
    mse_hybrid = mean_squared_error(y_test, y_pred_hybrid)
    mse_simple = mean_squared_error(y_test, y_pred_simple)
    mse_naive = mean_squared_error(y_test, y_pred_naive)

    print("\nBenchmarking Results (MSE):")
    print("Hybrid Model MSE:      {:.6f}".format(mse_hybrid))
    print("Simple LSTM Model MSE: {:.6f}".format(mse_simple))
    print("Naive Baseline MSE:    {:.6f}".format(mse_naive))

    # Plotting predictions
    plot_predictions(y_test, y_pred_hybrid, "Hybrid predictions (scaled)")

    # Plotting comparisons
    predictions = {
        "Hybrid Model": y_pred_hybrid.flatten(),
        "Simple Model": y_pred_simple.flatten(),
        "Naive Baseline": y_pred_naive.flatten(),
    }
    plot_comparison(
        y_test, predictions, title="Comparison of forecasting models (scaled)"
    )

    # Save models for backtesting
    hybrid_model.save("models/hybrid_model.keras")
    simple_model.save("models/simple_model.keras")
    print("Models saved to disk.")


if __name__ == "__main__":
    main()
