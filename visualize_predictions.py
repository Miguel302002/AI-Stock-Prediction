import numpy as np
import matplotlib.pyplot as plt

# Load predictions
y_test_lstm = np.load('lstm_y_test.npy')
y_pred_lstm = np.load('lstm_y_pred.npy')

y_test_rf = np.load('rf_y_test.npy')
y_pred_rf = np.load('rf_y_pred.npy')

# Plot actual vs. predicted for LSTM
plt.figure(figsize=(12, 6))
plt.plot(y_test_lstm, label='Actual Movement', marker='o')
plt.plot(y_pred_lstm, label='Predicted Movement (LSTM)', marker='x')
plt.title('LSTM Model: Actual vs. Predicted Stock Movements')
plt.xlabel('Test Sample Index')
plt.ylabel('Movement (1 = Up, 0 = Down)')
plt.legend()
plt.savefig('lstm_actual_vs_predicted.png')
plt.close()
print("LSTM actual vs. predicted plot saved as 'lstm_actual_vs_predicted.png'.")

# Plot actual vs. predicted for Random Forest
plt.figure(figsize=(12, 6))
plt.plot(y_test_rf, label='Actual Movement', marker='o')
plt.plot(y_pred_rf, label='Predicted Movement (Random Forest)', marker='x')
plt.title('Random Forest Model: Actual vs. Predicted Stock Movements')
plt.xlabel('Test Sample Index')
plt.ylabel('Movement (1 = Up, 0 = Down)')
plt.legend()
plt.savefig('rf_actual_vs_predicted.png')
plt.close()
print("Random Forest actual vs. predicted plot saved as 'rf_actual_vs_predicted.png'.")
