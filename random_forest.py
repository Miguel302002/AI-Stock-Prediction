# random_forest.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from evaluation import evaluate_model  # Import evaluation function

# 1. Load preprocessed data
data = pd.read_csv('processed_data.csv')

# 2. Define features and target
X = data[['Close', 'MA5', 'MA10', 'volatility', 'daily_return']]
y = (data['Close'].shift(-1) > data['Close']).astype(int)  # 1 if price goes up, 0 if it goes down

# Drop the last row to handle NaN in the target
X = X[:-1]
y = y[:-1]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy)

# Save accuracy to a text file
with open('random_forest_results.txt', 'w') as f:
    f.write(f"Random Forest Accuracy: {accuracy}\n")

# Evaluate and save confusion matrix plot and classification report
evaluate_model(y_test, y_pred, "Random Forest")
