import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directories
plots_dir = 'plots'
predictions_dir = 'predictions'
os.makedirs(plots_dir, exist_ok=True)

# Load preprocessed data for ticker info
data = pd.read_csv('processed_data/processed_data_pca.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

# Define time_steps consistent with model training
time_steps = 30

for ticker in data['Ticker'].unique():
    print(f"\nGenerating plots for {ticker}...")

    # Load predictions for LSTM
    lstm_y_test_file = os.path.join(predictions_dir, f'lstm_y_test_{ticker}.npy')
    lstm_y_pred_file = os.path.join(predictions_dir, f'lstm_y_pred_{ticker}.npy')
    
    if os.path.exists(lstm_y_test_file) and os.path.exists(lstm_y_pred_file):
        y_test = np.load(lstm_y_test_file)
        y_pred = np.load(lstm_y_pred_file)
        
        # Actual vs. Predicted Plot
        plt.figure(figsize=(14, 7))
        plt.plot(range(len(y_test)), y_test, label='Actual', marker='o', linestyle='-', markersize=4)
        plt.plot(range(len(y_pred)), y_pred, label='Predicted (LSTM)', marker='x', linestyle='--', markersize=4)
        plt.title(f'LSTM Model: Actual vs. Predicted Movements for {ticker}')
        plt.xlabel('Index')
        plt.ylabel('Movement (1 = Up, 0 = Down)')
        plt.legend()
        plt.tight_layout()
        lstm_plot_file = os.path.join(plots_dir, f'lstm_actual_vs_predicted_{ticker}.png')
        plt.savefig(lstm_plot_file)
        plt.close()
        print(f"LSTM Actual vs. Predicted plot for {ticker} saved as '{lstm_plot_file}'.")

    # Load predictions for Random Forest
    rf_y_test_file = os.path.join(predictions_dir, f'rf_y_test_{ticker}.npy')
    rf_y_pred_file = os.path.join(predictions_dir, f'rf_y_pred_{ticker}.npy')
    
    if os.path.exists(rf_y_test_file) and os.path.exists(rf_y_pred_file):
        y_test = np.load(rf_y_test_file)
        y_pred = np.load(rf_y_pred_file)

        # Actual vs. Predicted Plot
        plt.figure(figsize=(14, 7))
        plt.plot(range(len(y_test)), y_test, label='Actual', marker='o', linestyle='-', markersize=4)
        plt.plot(range(len(y_pred)), y_pred, label='Predicted (Random Forest)', marker='x', linestyle='--', markersize=4)
        plt.title(f'Random Forest Model: Actual vs. Predicted Movements for {ticker}')
        plt.xlabel('Index')
        plt.ylabel('Movement (1 = Up, 0 = Down)')
        plt.legend()
        plt.tight_layout()
        rf_plot_file = os.path.join(plots_dir, f'rf_actual_vs_predicted_{ticker}.png')
        plt.savefig(rf_plot_file)
        plt.close()
        print(f"Random Forest Actual vs. Predicted plot for {ticker} saved as '{rf_plot_file}'.")
