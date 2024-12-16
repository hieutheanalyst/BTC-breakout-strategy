import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_strategy(file_name, period=20, multiplier=2, x_bars=5, stop_loss=0.2):
    # Read in the data
    data = pd.read_csv(file_name)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    # Compute Bollinger Bands
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['STD'] = data['Close'].rolling(window=period).std()
    data['Upper Band'] = data['SMA'] + (multiplier * data['STD'])
    data['Lower Band'] = data['SMA'] - (multiplier * data['STD'])

    # Drop rows with NaN values
    data.dropna(inplace=True)

    # Add Buy and Sell Signals
    data['Buy Signal'] = data['Close'] > data['Upper Band']
    data['Sell Signal'] = data['Close'] < data['Lower Band']

    # Initialize variables
    capital = 10000
    position = 0
    entry_price = 0
    equity_curve = [capital]
    returns_series = [0]
    number_of_trades = 0

    # Iterate through the data to simulate the strategy
    for i in range(period, len(data)):
        current_date = data.index[i]
        if position == 0 and data['Buy Signal'].iloc[i]:  # Buy Signal
            position = 1
            entry_price = data['Close'].iloc[i]
            entry_date = current_date
            stop_loss_price = entry_price * (1 - stop_loss)
            number_of_trades += 1
        elif position == 1:
            # Check if stop loss is hit
            if data['Close'].iloc[i] <= stop_loss_price:
                exit_price = stop_loss_price
                position = 0
                returns = (exit_price - entry_price) / entry_price
                equity_curve.append(equity_curve[-1] * (1 + returns))
                returns_series.append(returns)
            # Check if x_bars have passed
            elif (current_date - entry_date).days >= x_bars:
                exit_price = data['Close'].iloc[i]
                position = 0
                returns = (exit_price - entry_price) / entry_price
                equity_curve.append(equity_curve[-1] * (1 + returns))
                returns_series.append(returns)

    # Calculate performance metrics
    returns_series = np.array(returns_series)
    equity_curve = np.array(equity_curve)
    cumulative_returns = (equity_curve / equity_curve[0] - 1) * 100  # Cumulative returns in percentage
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak * 100  # Drawdown in percentage

    return cumulative_returns, drawdowns, returns_series

# Usage
if __name__ == "__main__":
    cumulative_returns, drawdowns = run_strategy(file_name=r'btc_in_sample_data.csv', period=20, multiplier=2, x_bars=5, stop_loss=0.2)
    print(cumulative_returns)
