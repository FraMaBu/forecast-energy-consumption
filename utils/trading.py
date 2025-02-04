import numpy as np
import matplotlib.pyplot as plt


def backtest_strategy(
    actual_prices, predictions, initial_capital=1000.0, threshold=0.0
):
    """
    Backtests a simple trading strategy based on model predictions.

    Strategy:
      - For each trading day i, if the forecast for day i+1 (predictions[i]) is greater than
        the current day's price actual_prices[i] by at least `threshold` percent, go long.
      - Otherwise, remain in cash.
      - The trade is simulated by buying at actual_prices[i] and selling at actual_prices[i+1].

    Parameters:
      actual_prices (array-like): Array of actual prices of length n+1.
      predictions (array-like): Array of predicted next-day prices of length n.
      initial_capital (float): Starting capital.
      threshold (float): Minimum required percentage difference to trigger a trade.

    Returns:
      portfolio (np.array): Portfolio values at each time step (length n+1).
      signals (np.array): Trading signals for each day (1 for long, 0 for no trade).
    """
    n = len(predictions)
    portfolio = np.zeros(n + 1)
    portfolio[0] = initial_capital
    signals = np.zeros(n)

    for i in range(n):
        # If forecasted price exceeds current price by the threshold, take a long position.
        if predictions[i] > actual_prices[i] * (1 + threshold):
            signals[i] = 1
            ret = actual_prices[i + 1] / actual_prices[i]
        else:
            signals[i] = 0
            ret = 1.0  # No change (cash position)
        portfolio[i + 1] = portfolio[i] * ret

    return portfolio, signals


def plot_backtest(portfolio, title="Backtest Portfolio Value"):
    """
    Plots the portfolio value over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio, label="Portfolio Value")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.show()
