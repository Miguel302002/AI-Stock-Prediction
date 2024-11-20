results = {
    'Random Forest': {
        'Accuracy File': 'random_forest_results.txt',
        'Confusion Matrix': 'Random_Forest_confusion_matrix.png',
        'Classification Report': 'Random_Forest_classification_report.txt'
    },
    'LSTM': {
        'Accuracy File': 'lstm_results.txt',
        'Confusion Matrix': 'LSTM_confusion_matrix.png',
        'Classification Report': 'LSTM_classification_report.txt'
    }
}

with open('model_comparison.txt', 'w', encoding='utf-8') as f:
    f.write("Model Comparison Summary\n")
    f.write("=" * 50 + "\n\n")
    for model, paths in results.items():
        f.write(f"Model: {model}\n")
        f.write("=" * 50 + "\n")
        for key, path in paths.items():
            if path.endswith(".png"):
                f.write(f"{key}: {path} (image file saved)\n\n")
                continue
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                f.write(f"{key}:\n{content}\n")
                print(f"{model} - {key}:\n{content}\n")
            except FileNotFoundError:
                f.write(f"{key}: File not found.\n")
                print(f"{model} - {key}: File not found.\n")
        f.write("\n" + "-" * 50 + "\n\n")
