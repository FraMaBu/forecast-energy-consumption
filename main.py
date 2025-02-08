from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.data import load_power_data, scale_data, create_sliding_windows
from utils.model import (
    build_hybrid_model,
    build_LSTM_model,
    build_linear_regression_model,
)
from utils.train import train_model, evaluate_model, plot_comparison


# Parameters
WINDOW_SIZE = 168
TRAIN_SPLIT = 0.8
EPOCHS = 20
BATCH_SIZE = 32


def main():
    # Data loading and preprocessing
    # Load the power consumption data from the local file and resample to hourly means.
    power_data, dates = load_power_data(filepath="data/household_power_consumption.txt")
    power_scaled, scaler = scale_data(power_data)

    # Use a window size of 168 hours (i.e. one week) to capture both short-term and long-term patterns.
    window_size = WINDOW_SIZE
    num_fourier = 12
    X_seq, X_fourier, y = create_sliding_windows(
        power_scaled, dates, window_size=WINDOW_SIZE
    )

    # Train/Test split (80% train, 20% test)
    train_size = int(len(X_seq) * TRAIN_SPLIT)
    X_seq_train, X_seq_test = X_seq[:train_size], X_seq[train_size:]
    X_fourier_train, X_fourier_test = X_fourier[:train_size], X_fourier[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(
        f"Training dataset: {train_size} records. Test dataset: {len(y_test)} records."
    )

    # Linear regression baseline
    print("Building and training Linear Regression baseline...")
    lr_model = build_linear_regression_model(WINDOW_SIZE)
    train_model(
        lr_model,
        X_seq_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
    )
    lr_loss = evaluate_model(lr_model, X_seq_test, y_test)
    print("Linear Regression Model Test Loss:", lr_loss)
    y_pred_lr = lr_model.predict(X_seq_test)

    # Simple LSTM Model
    print("Building and training simple LSTM model on power consumption data...")
    simple_model = build_LSTM_model(window_size)
    train_model(
        simple_model,
        X_seq_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
    )
    simple_loss = evaluate_model(simple_model, X_seq_test, y_test)
    print("Simple LSTM Model Test Loss:", simple_loss)
    y_pred_simple = simple_model.predict(X_seq_test)

    # Hybrid Model
    print("Building and training hybrid model on power consumption data...")
    hybrid_model = build_hybrid_model(
        window_size, num_fourier, lstm_units=50, dense_units=20
    )
    train_model(
        hybrid_model,
        [X_seq_train, X_fourier_train],
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
    )
    hybrid_loss = evaluate_model(hybrid_model, [X_seq_test, X_fourier_test], y_test)
    print("Hybrid LSTM Model Test Loss:", hybrid_loss)
    y_pred_hybrid = hybrid_model.predict([X_seq_test, X_fourier_test])

    # Compare results
    # Inverse the scaling so that we can compare and plot in the original units.
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_lr_unscaled = scaler.inverse_transform(y_pred_lr.reshape(-1, 1)).flatten()
    y_pred_simple_unscaled = scaler.inverse_transform(
        y_pred_simple.reshape(-1, 1)
    ).flatten()
    y_pred_hybrid_unscaled = scaler.inverse_transform(
        y_pred_hybrid.reshape(-1, 1)
    ).flatten()

    mse_lr = mean_squared_error(y_test_unscaled, y_pred_lr_unscaled)
    mse_simple = mean_squared_error(y_test_unscaled, y_pred_simple_unscaled)
    mse_hybrid = mean_squared_error(y_test_unscaled, y_pred_hybrid_unscaled)

    mae_lr = mean_absolute_error(y_test_unscaled, y_pred_lr_unscaled)
    mae_simple = mean_absolute_error(y_test_unscaled, y_pred_simple_unscaled)
    mae_hybrid = mean_absolute_error(y_test_unscaled, y_pred_hybrid_unscaled)

    print("\nBenchmarking Results (on unscaled data):")
    print("Linear Regression - MSE: {:.6f}, MAE: {:.6f}".format(mse_lr, mae_lr))
    print("LSTM Model - MSE: {:.6f}, MAE: {:.6f}".format(mse_simple, mae_simple))
    print("Hybrid Model - MSE: {:.6f}, MAE: {:.6f}".format(mse_hybrid, mae_hybrid))

    # Plotting the comparison
    predictions = {
        "Hybrid Model": y_pred_hybrid_unscaled,
        "Simple Model": y_pred_simple_unscaled,
        "Linear Regression": y_pred_lr_unscaled,
    }

    plot_comparison(
        y_test_unscaled,
        predictions,
        title="Comparison of Model Predictions on Power Consumption Data (Unscaled)",
        xlabel="Time (hours)",
        ylabel="Global Active Power (kilowatts)",
        sample_rate=24,
    )

    # Save trained models if desired.
    hybrid_model.save("files/models/hybrid_model_power.keras")
    simple_model.save("files/models/simple_model_power.keras")
    print("Models saved to disk.")


if __name__ == "__main__":
    main()
