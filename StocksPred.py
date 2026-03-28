import numpy
import pandas
import matplotlib
import sklearn
import yfinance
import tensorflow

import yfinance as yf
import pandas as pd

ticker = input("Enter stock ticker (e.g., AAPL, INFY.NS, RELIANCE.NS): ")
start_date = input("Enter data collection start date (YYYY-MM-DD): ")
end_date = input("Enter data collection end date (YYYY-MM-DD): ")
# Fetch stock data
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    print("Invalid ticker or no data found!")
    exit()

# Show first 5 rows
print(data.head())

import matplotlib.pyplot as plt

# Plot closing price
plt.figure(figsize=(10,5))
plt.plot(data['Close'])
plt.title("Stock Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

print("Step 1: Starting.")
data = data[['Close']]

print("Step 2: Data selected.")
dataset = data.values

#Data converted into array.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

print("Step 3: Data Scaled.")
training_data_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[0:training_data_len]

print("Step 4: Training data ready.")
X_train = []
y_train = []

for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i])
    y_train.append(train_data[i])

#Loop done
import numpy as np
X_train, y_train = np.array(X_train), np.array(y_train)

#Converted to numpy.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print("Step 5: Data Reshaped.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Build model
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

print("Training started...")

model.fit(X_train, y_train, epochs=5, batch_size=32)

print("Training completed!")


# Prepare test data
test_data = scaled_data[training_data_len - 60:]

X_test = []
y_test = dataset[training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

import matplotlib.pyplot as plt

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(10,5))
plt.plot(train['Close'])
plt.plot(valid['Close'])
plt.plot(valid['Predictions'])

plt.title("Model Prediction vs Actual")
plt.xlabel("Date")
plt.ylabel("Price")

plt.legend(['Train', 'Actual', 'Predicted'])
plt.show()

future_days = 30
predictions = []

current_input = scaled_data[-60:]

for _ in range(future_days):
    current_input_reshaped = current_input.reshape(1, 60, 1)
    
    pred = model.predict(current_input_reshaped)
    predictions.append(pred[0][0])
    
    current_input = np.append(current_input[1:], pred)

# Convert back
predictions = np.array(predictions).reshape(-1,1)
predictions = scaler.inverse_transform(predictions)

print(predictions)

import matplotlib.pyplot as plt

plt.plot(predictions, label="Future Prices")
plt.legend()
plt.show()

import pandas as pd

future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)

plt.figure(figsize=(10,5))

plt.plot(data.index[-100:], data['Close'][-100:], label=f"{ticker} Actual Price")
plt.plot(future_dates, predictions, label="Future Prediction")

plt.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")

print("\n")
plt.show()
