# lstm_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from sklearn.preprocessing import MinMaxScaler
from evaluation import evaluate_model  # Import evaluation function

# 1. Load preprocessed data
data = pd.read_csv('processed_data.csv')

# 2. Define features and target
X = data[['Close', 'MA5', 'MA10', 'volatility', 'daily_return']]
y = (data['Close'].shift(-1) > data['Close']).astype(int)  # 1 if price goes up, 0 if it goes down

# Drop the last row to handle NaN in the target
X = X[:-1]
y = y[:-1]

# 3. Scale features for LSTM
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for LSTM: (samples, time_steps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))  # Binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# 7. Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# 8. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"LSTM Model Accuracy: {accuracy}")

# Save accuracy to a text file
with open('lstm_results.txt', 'w') as f:
    f.write(f"LSTM Model Accuracy: {accuracy}\n")

# Evaluate and save confusion matrix plot and classification report
evaluate_model(y_test, y_pred, "LSTM")
