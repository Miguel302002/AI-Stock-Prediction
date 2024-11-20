import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from evaluation import evaluate_model
import warnings

# Load preprocessed data
data = pd.read_csv('processed_data.csv')
print("Preprocessed data loaded.")

# Ensure 'Date' is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Define features and target
features = ['Close', 'MA5', 'MA10', 'volatility']
data['Target'] = (data.groupby('Ticker')['Close'].shift(-1) > data['Close']).astype(int)

# Avoid using groupby().apply() by dropping last row of each group directly
data = data.sort_values(by=['Ticker', 'Date'])
grouped = data.groupby('Ticker')
indices_to_drop = grouped.tail(1).index
data = data.drop(indices_to_drop).reset_index(drop=True)
print("Target variable created and last rows dropped.")

X = data[features]
y = data['Target']

# Split data into training and testing sets (maintaining chronological order)
split_date = '2022-12-31'  # Same split date as in preprocessing
train_data = data[data['Date'] <= split_date]
test_data = data[data['Date'] > split_date]

X_train = train_data[features]
y_train = train_data['Target']
X_test = test_data[features]
y_test = test_data['Target']

# Check class distribution
print("Training set class distribution:")
print(y_train.value_counts())
print("Testing set class distribution:")
print(y_test.value_counts())

# Before training the model
# Check for NaNs or Infs in X_train and y_train
if X_train.isnull().values.any() or not np.isfinite(X_train.values).all():
    print("Warning: X_train contains NaN or infinite values.")
    # Optionally, handle or remove NaNs/Infs
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]

if y_train.isnull().values.any() or not np.isfinite(y_train.values).all():
    print("Warning: y_train contains NaN or infinite values.")
    # Optionally, handle or remove NaNs/Infs
    y_train = y_train.dropna()
    X_train = X_train.loc[y_train.index]

# Ensure X_train and y_train have matching indices
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Expanded hyperparameter grid
param_distributions = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Suppress any convergence warnings during hyperparameter tuning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

    # Use RandomizedSearchCV for broader search
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=50,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    print("Starting RandomizedSearchCV...")
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV completed.")

# Train Random Forest with best parameters
model = random_search.best_estimator_
print(f"Best parameters found: {random_search.best_params_}")
model.fit(X_train, y_train)
print("Random Forest model trained.")

# Make predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")

# Save accuracy to a text file
with open('random_forest_results.txt', 'w') as f:
    f.write(f"Random Forest Accuracy: {accuracy}\n")

# Save y_test and y_pred
np.save('rf_y_test.npy', y_test)
np.save('rf_y_pred.npy', y_pred)

# Evaluate and save confusion matrix and classification report
evaluate_model(y_test, y_pred, y_pred_prob, "Random_Forest")

# Feature Importance Analysis
import matplotlib.pyplot as plt

feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)[::-1]

# Plot the feature importances
plt.figure()
plt.title("Feature Importances - Random Forest")
plt.bar(range(len(features)), feature_importances[indices], align='center')
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig('random_forest_feature_importance.png')
plt.close()
print("Feature importance plot saved as 'random_forest_feature_importance.png'.")

# Save the model
import joblib
joblib.dump(model, 'random_forest_model.pkl')
print("Random Forest model saved as 'random_forest_model.pkl'.")
