import os

# Define paths to results for comparison
stocks = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA']
results = {
    'Random Forest': {
        stock: {
            'Accuracy File': f'evaluation_results/random_forest_results_{stock}.txt',
            'Confusion Matrix': f'plots/Random_Forest_{stock}_confusion_matrix.png',
            'Classification Report': f'plots/Random_Forest_{stock}_classification_report.txt',
            'Feature Importance': f'plots/random_forest_feature_importance_{stock}.png',
            'ROC Curve': f'plots/Random_Forest_{stock}_roc_curve.png',
            'Actual vs Predicted': f'plots/rf_actual_vs_predicted_{stock}.png'
        } for stock in stocks
    },
    'LSTM': {
        stock: {
            'Confusion Matrix': f'plots/LSTM_{stock}_confusion_matrix.png',
            'Classification Report': f'plots/LSTM_{stock}_classification_report.txt',
            'ROC Curve': f'plots/LSTM_{stock}_roc_curve.png',
            'Actual vs Predicted': f'plots/lstm_actual_vs_predicted_{stock}.png'
        } for stock in stocks
    }
}

# Create output directory for comparison
comparison_dir = 'comparison'
os.makedirs(comparison_dir, exist_ok=True)

# File to store the comparison results
comparison_file = os.path.join(comparison_dir, 'model_comparison.txt')

# Write model comparison summary
with open(comparison_file, 'w', encoding='utf-8') as f:
    f.write("Model Comparison Summary\n")
    f.write("=" * 50 + "\n\n")
    
    for model, stock_results in results.items():
        f.write(f"Model: {model}\n")
        f.write("=" * 50 + "\n")
        for stock, paths in stock_results.items():
            f.write(f"Stock: {stock}\n")
            f.write("-" * 50 + "\n")
            for key, path in paths.items():
                if path.endswith(".png"):
                    # Record image file paths
                    f.write(f"{key}: {path} (image file saved)\n")
                    continue
                try:
                    # Try to open and read text-based files
                    with open(path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    f.write(f"{key}:\n{content}\n")
                    print(f"{model} - {stock} - {key}:\n{content}\n")
                except FileNotFoundError:
                    # Handle missing files gracefully
                    f.write(f"{key}: File not found.\n")
                    print(f"{model} - {stock} - {key}: File not found.\n")
                except Exception as e:
                    # Catch other potential errors
                    f.write(f"{key}: Error reading file: {e}\n")
                    print(f"{model} - {stock} - {key}: Error reading file: {e}\n")
            f.write("\n" + "-" * 50 + "\n")
        f.write("\n" + "=" * 50 + "\n\n")

print(f"Model comparison results saved in '{comparison_file}'.")
