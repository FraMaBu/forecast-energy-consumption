import numpy as np
import matplotlib.pyplot as plt


def train_model(
    model, X_train, y_train, epochs=20, batch_size=32, validation_split=0.1
):
    """
    Trains a Keras model.
    """
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
    )
    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a model on test data.
    """
    loss = model.evaluate(X_test, y_test)
    return loss


def plot_comparison(
    y_test,
    predictions_dict,
    title="Comparison of Forecasting Models",
    xlabel="Time step",
    ylabel="Scaled Value",
    sample_rate=24,  # Plot every 24th hour by default
):
    """
    Plots a comparison of multiple models’ predictions.

    Parameters:
      - y_test: Array-like ground truth values.
      - predictions_dict: Dictionary mapping model names to prediction arrays.
      - sample_rate: Integer. Only every sample_rate-th data point will be plotted to avoid overcrowding.
      - title, xlabel, ylabel: Plot labeling.

    The function downsamples the data for plotting if the full hourly data is too dense.
    """
    # Create an array of indices that will be used for plotting.
    indices = np.arange(0, len(y_test), sample_rate)

    plt.figure(figsize=(14, 7))
    plt.plot(indices, y_test[indices], label="True Value", linestyle="--")

    for label, pred in predictions_dict.items():
        plt.plot(indices, pred[indices], label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
