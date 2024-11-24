import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np

plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

def evaluate_model(y_test, y_pred, y_pred_prob, model_name):
    """
    Evaluate model performance and generate metrics, confusion matrix, and ROC curve.
    """
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    cm_file = os.path.join(plots_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_file)
    plt.close()
    print(f"Confusion matrix saved as '{cm_file}'.")

    # Classification Report
    cr = classification_report(y_test, y_pred)
    cr_file = os.path.join(plots_dir, f"{model_name}_classification_report.txt")
    with open(cr_file, 'w') as f:
        f.write(cr)
    print(f"Classification Report for {model_name}:\n\n{cr}")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    roc_file = os.path.join(plots_dir, f"{model_name}_roc_curve.png")
    plt.savefig(roc_file)
    plt.close()
    print(f"ROC curve saved as '{roc_file}'.")
