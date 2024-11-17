# evaluation.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def evaluate_model(y_test, y_pred, model_name):
    """
    Evaluates the model using a confusion matrix and classification report.
    Generates a confusion matrix plot and saves it as an image file.
    
    Parameters:
    - y_test: True labels
    - y_pred: Predicted labels
    - model_name: Name of the model (for display purposes)
    """
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", colorbar=False, ax=ax)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # Save the plot as an image file
    plot_filename = f"{model_name}_confusion_matrix.png"
    plt.savefig(plot_filename)
    plt.close()

    # Generate classification report
    report = classification_report(y_test, y_pred)
    print(f"Classification Report for {model_name}:\n")
    print(report)

    # Save classification report to a text file
    report_filename = f"{model_name}_classification_report.txt"
    with open(report_filename, 'w') as f:
        f.write(f"Classification Report for {model_name}:\n\n")
        f.write(report)
