import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import ta

# 1. Load the dataset
data = pd.read_csv('major-tech-stock-2019-2024.csv')
print("Data loaded successfully.")

# Ensure 'Date' is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# 2. Handle missing data
data = data.sort_values(by=['Ticker', 'Date'])

# Fill missing values
data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = data.groupby('Ticker')[
    ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
].transform(lambda x: x.ffill().bfill())
print("Missing data handled.")

# 3. Feature engineering

def compute_technical_indicators(group):
    group = group.copy()
    # RSI
    group['RSI'] = ta.momentum.RSIIndicator(close=group['Close']).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=group['Close'])
    group['MACD'] = macd.macd()
    group['MACD_diff'] = macd.macd_diff()
    group['MACD_signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=group['Close'])
    group['BB_high'] = bb.bollinger_hband()
    group['BB_low'] = bb.bollinger_lband()
    
    # Daily Return
    group['daily_return'] = group['Close'].pct_change()
    
    # Moving Averages
    group['MA5'] = group['Close'].rolling(window=5, min_periods=1).mean()
    group['MA10'] = group['Close'].rolling(window=10, min_periods=1).mean()
    
    # Volatility
    group['volatility'] = group['daily_return'].rolling(window=10, min_periods=1).std()
    
    return group

# Apply the function to each group
data = data.groupby('Ticker', group_keys=False).apply(compute_technical_indicators).reset_index(drop=True)
print("Technical indicators computed.")

# Lag Features
lag_days = 5
for lag in range(1, lag_days + 1):
    data[f'Close_lag_{lag}'] = data.groupby('Ticker')['Close'].shift(lag)
    data[f'Volume_lag_{lag}'] = data.groupby('Ticker')['Volume'].shift(lag)

# Time-Based Features
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
data['Quarter'] = data['Date'].dt.quarter

# Remove rows with NaN values due to new features
data = data.dropna().reset_index(drop=True)
print("Feature engineering completed.")

# 4. Combine features for PCA
features_for_pca = [
    'Close', 'MA5', 'MA10', 'volatility', 'MACD', 'MACD_diff', 'MACD_signal',
    'RSI', 'BB_high', 'BB_low'
] + [f'Close_lag_{lag}' for lag in range(1, lag_days + 1)] + \
    [f'Volume_lag_{lag}' for lag in range(1, lag_days + 1)] + ['Volume']

# Ensure all features are numeric and handle any missing values
data[features_for_pca] = data[features_for_pca].fillna(0)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features_for_pca])

# Apply PCA to retain 95% variance
pca = PCA(n_components=0.95, random_state=42)
principal_components = pca.fit_transform(scaled_features)

# Create a DataFrame with principal components
pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
pca_df = pd.DataFrame(data=principal_components, columns=pca_columns)

# Combine PCA components with other relevant columns
data.reset_index(drop=True, inplace=True)
processed_data = pd.concat([data[['Date', 'Ticker']], pca_df], axis=1)

# 5. Save the preprocessed data
processed_data.to_csv('processed_data_pca.csv', index=False)
print("Preprocessed data with PCA saved as 'processed_data_pca.csv'.")
