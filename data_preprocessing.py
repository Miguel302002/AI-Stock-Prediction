import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Load the dataset
data = pd.read_csv('major-tech-stock-2019-2024.csv')
print("Data loaded successfully.")

# Ensure 'Date' is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# 2. Explore the data
print("Dataset shape:", data.shape)
print("Data types:\n", data.dtypes)
print("First few rows:\n", data.head())
print("Summary statistics:\n", data.describe())

# 3. Handle missing data
data = data.sort_values(by=['Ticker', 'Date'])

# Avoid using groupby().apply() to eliminate the DeprecationWarning
# Use groupby() with transform to fill missing values
data[[
    'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
]] = data.groupby('Ticker')[[
    'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
]].transform(lambda x: x.ffill().bfill())
print("Missing data handled.")

# 4. Feature engineering
data['daily_return'] = data.groupby('Ticker')['Close'].pct_change()

# Use rolling with min_periods=1 to handle cases with fewer data points
data['MA5'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
data['MA10'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
data['volatility'] = data.groupby('Ticker')['daily_return'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
print("Feature engineering completed.")

# Remove initial rows with NaN values due to rolling calculations
data = data.dropna(subset=['daily_return', 'MA5', 'MA10', 'volatility'])
print("NaN values from rolling calculations removed.")

# 5. Split data into training and testing sets (maintaining chronological order)
split_date = '2022-12-31'  # Adjust based on your dataset
train_data = data[data['Date'] <= split_date]
test_data = data[data['Date'] > split_date]
print("Data split into training and testing sets.")

# 6. Scale/normalize features
features_to_scale = ['Close', 'MA5', 'MA10', 'volatility']
scaler = StandardScaler()

# Use .loc to avoid SettingWithCopyWarning
train_data.loc[:, features_to_scale] = scaler.fit_transform(train_data[features_to_scale])
test_data.loc[:, features_to_scale] = scaler.transform(test_data[features_to_scale])
print("Features scaled.")

# 7. Save the preprocessed data
processed_data = pd.concat([train_data, test_data], ignore_index=True)
processed_data.to_csv('processed_data.csv', index=False)
print("Preprocessed data saved as 'processed_data.csv'.")
