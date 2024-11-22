import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load the PCA-processed data
data = pd.read_csv('processed_data_pca.csv')
print("PCA-processed data loaded successfully.")

# 2. Add target variable
# Assuming the target is a binary classification based on the next day's 'Close' value
original_data = pd.read_csv('major-tech-stock-2019-2024.csv')
original_data['Target'] = (original_data.groupby('Ticker')['Close'].shift(-1) > original_data['Close']).astype(int)

# Merge Target with PCA data
data = pd.merge(data, original_data[['Date', 'Ticker', 'Target']], on=['Date', 'Ticker'])
data = data.dropna()  # Remove any rows with missing target values
print("Target variable merged successfully.")

# 3. Group by Ticker for per-company analysis
results = []
for company, group_data in data.groupby('Ticker'):
    print(f"\nProcessing SVM analysis for company: {company}")

    # 4. Split data into features and target
    X = group_data.drop(columns=['Date', 'Ticker', 'Target'])
    y = group_data['Target']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Train the SVM model
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)

    # 6. Make predictions
    y_pred = svm_model.predict(X_test)

    # 7. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"Model Accuracy for {company}: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(classification_rep)

    # Plot and save the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {company}')
    plt.savefig(f'confusion_matrix_{company}.png')
    plt.close()

    # Store results for this company
    results.append({
        'Company': company,
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
        'Classification Report': classification_rep
    })

# 8. Save results to a file
results_df = pd.DataFrame([{key: val for key, val in result.items() if key != 'Confusion Matrix'} for result in results])
results_df.to_csv('svm_results_per_company.csv', index=False)
print("\nSVM results per company saved to 'svm_results_per_company.csv'.")
