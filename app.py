import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Title of the web app
st.title("Stock Price Prediction")

# Sidebar for user inputs
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")  # Default: Apple
forecast_days = st.sidebar.slider("Days to forecast", 1, 60, 30)  # Forecast for 30 days by default

# Fetch historical data using yfinance
def load_data(ticker):
    data = yf.download(ticker, period="5y", interval="1d")
    return data

# Preprocess the data to create features for the model
def preprocess_data(data):
    data['Date'] = pd.to_datetime(data.index)
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Weekday'] = data['Date'].dt.weekday

    # Lag features: Use previous day's Close price
    data['Lag1'] = data['Close'].shift(1)
    data['Lag2'] = data['Close'].shift(2)
    data['Lag3'] = data['Close'].shift(3)

    # Drop NaN values that appear due to lag features
    data = data.dropna()

    return data

# Train model and make predictions
def train_predict_model(data, forecast_days):
    X = data[['Day', 'Month', 'Year', 'Weekday', 'Lag1', 'Lag2', 'Lag3']]
    y = data['Close']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Predict future prices
    last_data = data.tail(1)[['Day', 'Month', 'Year', 'Weekday', 'Lag1', 'Lag2', 'Lag3']].values
    future_predictions = []
    for _ in range(forecast_days):
        prediction = model.predict(last_data)[0]
        future_predictions.append(prediction)

        # Update the input for the next prediction
        last_data[0, 4:] = np.roll(last_data[0, 4:], shift=-1)  # Roll the lag features
        last_data[0, 4] = prediction  # Update the 'Lag1' with the new prediction

    return future_predictions, mae

# Display historical stock data and forecast
def plot_data(data, future_predictions, forecast_days):
    # Plot the historical data
    st.subheader("Historical Stock Data")
    st.write(data.tail(20))

    # Plot the prediction graph
    st.subheader(f"Stock Price Forecast for the Next {forecast_days} Days")
    forecast_dates = pd.date_range(start=data.index[-1], periods=forecast_days + 1, freq='D')[1:]
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label="Historical Prices")
    plt.plot(forecast_dates, future_predictions, label="Predicted Prices", linestyle='--', color='r')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.legend()
    st.pyplot(plt)

# Main execution
if ticker:
    st.write(f"**Fetching data for {ticker}...**")
    data = load_data(ticker)
    processed_data = preprocess_data(data)

    st.write(f"**Training model for {ticker}...**")
    future_predictions, mae = train_predict_model(processed_data, forecast_days)

    st.write(f"**Model Evaluation:** Mean Absolute Error: {mae:.2f}")
    plot_data(data, future_predictions, forecast_days)
