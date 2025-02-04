import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def train_model(model, X_train, y_train, epochs=20, batch_size=32, validation_split=0.1):
    """
    Trains a Keras model.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a model on test data.
    """
    loss = model.evaluate(X_test, y_test)
    return loss

def plot_predictions(y_test, y_pred, title="Model Prediction", xlabel="Time step", ylabel="Scaled Price"):
    """
    Plots the true vs. predicted prices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="True Price (scaled)")
    plt.plot(y_pred, label="Predicted Price (scaled)")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_comparison(y_test, predictions_dict, title="Comparison of Forecasting Models"):
    """
    Plots a comparison of multiple models’ predictions.
    
    predictions_dict should be a dictionary mapping model names to predictions.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label="True Price (scaled)", linestyle="--")
    for label, pred in predictions_dict.items():
        plt.plot(pred, label=label)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Scaled Price")
    plt.legend()
    plt.show()