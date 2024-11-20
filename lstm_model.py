import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow messages

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from evaluation import evaluate_model

# Load preprocessed data
data = pd.read_csv('processed_data.csv')
print("Preprocessed data loaded.")

# Ensure 'Date' is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Define features
features = ['Close', 'MA5', 'MA10', 'volatility']

# Define the target variable: next day's movement (1 for up, 0 for down)
data = data.sort_values(by=['Ticker', 'Date'])
data['Target'] = (data.groupby('Ticker')['Close'].shift(-1) > data['Close']).astype(int)

# Drop the last row of each group with NaN target
grouped = data.groupby('Ticker')
indices_to_drop = grouped.tail(1).index
data = data.drop(indices_to_drop).reset_index(drop=True)
print("Target variable created and last rows dropped.")

# Parameters
time_steps = 20  # Experiment with different values
X = []
y = []

# Create sequences for each ticker separately
for ticker in data['Ticker'].unique():
    ticker_data = data[data['Ticker'] == ticker]
    scaled_features = ticker_data[features].values
    targets = ticker_data['Target'].values

    # Create sequences
    for i in range(time_steps, len(scaled_features)):
        X.append(scaled_features[i - time_steps:i])
        y.append(targets[i])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets (maintaining temporal order)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Check class distribution
print("Training set class distribution:")
print(pd.Series(y_train).value_counts())
print("Testing set class distribution:")
print(pd.Series(y_test).value_counts())

# Calculate class weights to handle class imbalance
unique_classes = np.unique(y_train)
class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=y_train)
class_weights = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights_array)}
print("Class weights:", class_weights)

# Build LSTM model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Increased patience to allow more epochs
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=0  # Suppress training output
)

# Print training history
print("Training history:")
for key in history.history.keys():
    print(f"{key}: {history.history[key]}")

# Make predictions
y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"LSTM Model Loss: {loss}")
print(f"LSTM Model Accuracy: {accuracy}")

# Save accuracy to a text file
with open('lstm_results.txt', 'w') as f:
    f.write(f"LSTM Model Loss: {loss}\n")
    f.write(f"LSTM Model Accuracy: {accuracy}\n")

# Save y_test and y_pred
np.save('lstm_y_test.npy', y_test)
np.save('lstm_y_pred.npy', y_pred)

# Evaluate and save confusion matrix and classification report
evaluate_model(y_test, y_pred, y_pred_prob, "LSTM")

# Save the model
model.save('lstm_model.keras')
print("LSTM model saved as 'lstm_model.keras'.")
