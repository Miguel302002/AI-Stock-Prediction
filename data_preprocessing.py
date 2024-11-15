# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np

# 1. Load the dataset
data = pd.read_csv('major-tech-stock-2019-2024.csv')  # Replace with your actual dataset path
print("Data loaded successfully.")

# 2. Explore the data
print("Dataset shape:", data.shape)
print("Data types:\n", data.dtypes)
print("First few rows:\n", data.head())
print("Summary statistics:\n", data.describe())

# 3. Handle missing data
data = data.dropna()  # Or you can use fillna(data.mean()) if that’s your strategy
print("Missing data handled.")

# 4. Handle outliers (optional)
data = data[(np.abs(stats.zscore(data.select_dtypes(include=[np.number]))) < 3).all(axis=1)]
print("Outliers handled.")

# 5. Feature engineering
data['daily_return'] = data['Close'].pct_change()
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA10'] = data['Close'].rolling(window=10).mean()
data['volatility'] = data['daily_return'].rolling(window=10).std()
print("Feature engineering completed.")

# 6. Scale/normalize features
scaler = StandardScaler()
data[['Close', 'MA5', 'MA10', 'volatility']] = scaler.fit_transform(data[['Close', 'MA5', 'MA10', 'volatility']])
print("Features scaled.")

# 7. Split data (save preprocessed data instead of split if you only need preprocessed output here)
train_data = data.iloc[:int(0.8 * len(data))]
test_data = data.iloc[int(0.8 * len(data)):]
print("Data split into training and testing sets.")

# 8. Save the preprocessed data
data.to_csv('processed_data.csv', index=False)
print("Preprocessed data saved as 'processed_data.csv'.")
