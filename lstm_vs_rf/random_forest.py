import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight  # Missing import added
from evaluation import evaluate_model
import joblib
import matplotlib.pyplot as plt

# Create output directories
models_dir = 'models'
evaluation_dir = 'evaluation_results'
plots_dir = 'plots'
logs_dir = 'logs'
predictions_dir = 'predictions'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(evaluation_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

# Redirect console output to a log file
log_file = os.path.join(logs_dir, 'random_forest.log')
sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

# Load preprocessed data
data = pd.read_csv('processed_data/processed_data_pca.csv')
print("Preprocessed data loaded.")

# Ensure 'Date' is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Define features and target
pca_columns = [col for col in data.columns if 'PC' in col]
features = pca_columns

# Define target
data['Price'] = data['PC1']
data['Price_next'] = data.groupby('Ticker')['Price'].shift(-1)
data['Price_change'] = data['Price_next'] - data['Price']
data['Target'] = (data['Price_change'] > 0).astype(int)
data = data.dropna(subset=['Target']).reset_index(drop=True)
print("Target variable created.")

# Loop over each company
for ticker in data['Ticker'].unique():
    print(f"\nTraining Random Forest model for {ticker}...")
    ticker_data = data[data['Ticker'] == ticker].reset_index(drop=True)
    
    X = ticker_data[features].values
    y = ticker_data['Target'].values

    if len(X) == 0:
        print(f"Not enough data for {ticker}. Skipping...")
        continue
    
    # Split data into training and testing sets
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Balanced class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print("Class weights:", class_weights_dict)
    
    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Hyperparameter grid
    param_distributions = {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
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
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy for {ticker}: {accuracy}")
    
    # Save accuracy
    accuracy_file = os.path.join(evaluation_dir, f'random_forest_results_{ticker}.txt')
    with open(accuracy_file, 'w') as f:
        f.write(f"Random Forest Accuracy for {ticker}: {accuracy}\n")
    
    # Save predictions
    np.save(os.path.join(predictions_dir, f'rf_y_test_{ticker}.npy'), y_test)
    np.save(os.path.join(predictions_dir, f'rf_y_pred_{ticker}.npy'), y_pred)
    print(f"Predictions for {ticker} saved in '{predictions_dir}' directory.")
    
    # Evaluate and save confusion matrix and classification report
    evaluate_model(y_test, y_pred, y_pred_prob, f"Random_Forest_{ticker}")
    
    # Feature Importance Analysis
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    
    # Save feature importances
    feature_importance_file = os.path.join(evaluation_dir, f'rf_feature_importance_{ticker}.csv')
    pd.DataFrame({'Feature': features, 'Importance': feature_importances}).to_csv(feature_importance_file, index=False)
    print(f"Feature importances for {ticker} saved as '{feature_importance_file}'.")
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances - Random Forest for {ticker}")
    plt.bar(range(len(features)), feature_importances[indices], align='center')
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    feature_importance_plot = os.path.join(plots_dir, f'random_forest_feature_importance_{ticker}.png')
    plt.savefig(feature_importance_plot)
    plt.close()
    print(f"Feature importance plot for {ticker} saved as '{feature_importance_plot}'.")
    
    # Save the model
    model_file = os.path.join(models_dir, f'random_forest_model_{ticker}.pkl')
    joblib.dump(model, model_file)
    print(f"Random Forest model for {ticker} saved as '{model_file}'.")

# Close the log file
sys.stdout.close()
