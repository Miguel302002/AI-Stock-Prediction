import os
import sys
import io
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from evaluation import evaluate_model

# Ensure UTF-8 encoding for all outputs
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info/warning/error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Log file setup
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, 'lstm_model.log')

# Redirect output to the log file
log = open(log_file, 'w', encoding='utf-8', errors='replace')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = sys.stdout  # Synchronize stderr with stdout

# Create necessary directories
models_dir = 'models'
evaluation_dir = 'evaluation_results'
predictions_dir = 'predictions'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(evaluation_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

# Load preprocessed data
data = pd.read_csv('processed_data/processed_data_pca.csv')
print("Preprocessed data with PCA loaded.")

# Ensure 'Date' is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Define PCA features and target variable
pca_columns = [col for col in data.columns if 'PC' in col]
data = data.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
data['Price'] = data['PC1']
data['Price_next'] = data.groupby('Ticker')['Price'].shift(-1)
data['Price_change'] = data['Price_next'] - data['Price']
data['Target'] = (data['Price_change'] > 0).astype(int)
data = data.dropna(subset=['Target']).reset_index(drop=True)
print("Target variable created.")

# LSTM parameters
time_steps = 30  # Sequence length
batch_size = 16
epochs = 50

# Loop over each ticker
for ticker in data['Ticker'].unique():
    print(f"\nTraining LSTM model for {ticker}...")
    ticker_data = data[data['Ticker'] == ticker].reset_index(drop=True)
    features = ticker_data[pca_columns].values
    targets = ticker_data['Target'].values

    # Create time series sequences
    X, y = [], []
    for i in range(time_steps, len(features)):
        X.append(features[i - time_steps:i])
        y.append(targets[i])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print(f"Not enough data for {ticker}. Skipping...")
        continue

    # Split into training and testing sets
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Handle class imbalance with class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"Class weights for {ticker}: {class_weights_dict}")

    # Build the LSTM model
    model = Sequential([
        Bidirectional(LSTM(128, activation='relu', return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.4),
        Bidirectional(LSTM(64, activation='relu')),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights_dict,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Save training history
    history_file = os.path.join(logs_dir, f'lstm_history_{ticker}.json')
    with open(history_file, 'w') as f:
        json.dump(history.history, f)
    print(f"Training history saved for {ticker} as '{history_file}'.")

    # Evaluate the model
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"LSTM Model Loss for {ticker}: {loss}")
    print(f"LSTM Model Accuracy for {ticker}: {accuracy}")

    # Save predictions
    np.save(os.path.join(predictions_dir, f'lstm_y_test_{ticker}.npy'), y_test)
    np.save(os.path.join(predictions_dir, f'lstm_y_pred_{ticker}.npy'), y_pred)
    print(f"Predictions for {ticker} saved in '{predictions_dir}' directory.")

    # Generate evaluation metrics
    evaluate_model(y_test, y_pred, y_pred_prob, f"LSTM_{ticker}")

    # Save the model
    model_file = os.path.join(models_dir, f'lstm_model_{ticker}.keras')
    model.save(model_file)
    print(f"LSTM model for {ticker} saved as '{model_file}'.")

# Close the log file
sys.stdout.close()
log.close()
