import subprocess
import sys
import os

# List of scripts to run in order
scripts = [
    'data_preprocessing.py',
    'random_forest.py',
    'lstm_model.py',
    'visualize_predictions.py',
    'model_comparison.py'
]

for script in scripts:
    print(f"\nRunning {script}...")
    # Run the script and capture output
    process = subprocess.run(
        [sys.executable, script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',  # Ensure UTF-8 encoding
        errors='replace',   # Replace undecodable characters
        env=dict(os.environ, PYTHONIOENCODING='utf-8')  # Set environment variable
    )
    # Print the script's output
    print(process.stdout)
    # Print any errors
    if process.stderr:
        print(f"Errors in {script}:")
        print(process.stderr)
    # Check if the script exited with an error code
    if process.returncode != 0:
        print(f"{script} exited with an error code {process.returncode}. Stopping execution.")
        break
    print(f"Finished running {script}.\n{'-'*50}")
