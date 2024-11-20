import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_auc_score,
)
import numpy as np

def evaluate_model(y_test, y_pred, y_pred_prob, model_name):
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", colorbar=False, ax=ax)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved as '{model_name}_confusion_matrix.png'.")

    # Classification report
    report = classification_report(y_test, y_pred, zero_division=1)

    # Additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    # Save metrics to the report
    with open(f"{model_name}_classification_report.txt", 'w', encoding='utf-8') as f:
        f.write(f"Classification Report for {model_name}:\n\n")
        f.write(report)
        f.write(f"\nAccuracy: {accuracy}\n")
        f.write(f"ROC AUC Score: {roc_auc}\n")

    # Print classification report to console
    print(f"Classification Report for {model_name}:\n")
    print(report)
    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC Score: {roc_auc}")
