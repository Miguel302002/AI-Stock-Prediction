import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load preprocessed data
data = pd.read_csv('processed_data_pca.csv')
print("Preprocessed data loaded.")

# Ensure 'Date' is in datetime format and sort by 'Ticker' and 'Date'
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

# Define features and target
sequence_length = 10  # Use 10 past timesteps for prediction
features = [col for col in data.columns if col.startswith('PC')]
target = 'PC1'  # Predict the first principal component as an example

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[features])
data[features] = scaled_features

# Prepare sequences and targets for RNN grouped by Ticker
X, y = [], []
for ticker in data['Ticker'].unique():
    ticker_data = data[data['Ticker'] == ticker]
    scaled_features_ticker = ticker_data[features].values
    for i in range(sequence_length, len(scaled_features_ticker)):
        X.append(scaled_features_ticker[i-sequence_length:i])  # Sequence of past 10 timesteps
        y.append(scaled_features_ticker[i][features.index(target)])  # Predict PC1

X, y = np.array(X), np.array(y)

# Convert target to binary classification (e.g., threshold at 0.5)
y = (y > 0.5).astype(int)

# Split data into training and testing sets (80/20 split)
split_index = int(0.8 * len(X))  # 80% for training, 20% for testing
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(sequence_length, len(features))),
    tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'),
    tf.keras.layers.LSTM(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=1)

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"RNN Model Accuracy: {accuracy}")

# Save accuracy to a text file
with open('rnn_results.txt', 'w') as f:
    f.write(f"RNN Model Accuracy: {accuracy}\n")

# Save y_test and y_pred
np.save('rnn_y_test.npy', y_test)
np.save('rnn_y_pred.npy', y_pred)

# Generate confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - RNN")
plt.savefig("rnn_confusion_matrix.png")
plt.close()
print("Confusion matrix saved as 'rnn_confusion_matrix.png'.")

# Save classification report
with open('rnn_classification_report.txt', 'w') as f:
    f.write("Classification Report - RNN\n")
    f.write(class_report)
print("Classification report saved as 'rnn_classification_report.txt'.")

# Save the trained model
model.save('rnn_stock_prediction_model.h5')
print("RNN model saved as 'rnn_stock_prediction_model.h5'.")
