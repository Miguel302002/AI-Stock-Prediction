import subprocess
import sys
import os
import shutil

# Define output directories
output_dirs = [
    'processed_data',
    'models',
    'evaluation_results',
    'logs',
    'predictions',
    'plots'
]

# Function to clear and recreate directories
def reset_output_dirs(dirs):
    for directory in dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)  # Delete the directory and its contents
        os.makedirs(directory)  # Recreate the directory
        print(f"Reset directory: {directory}")

# Reset output directories
print("Resetting output directories...\n" + "-" * 50)
reset_output_dirs(output_dirs)

# Create logs directory
logs_dir = 'logs'

# List of scripts to run in order
scripts = [
    'data_preprocessing.py',
    'random_forest.py',
    'lstm_model.py',
    'visualize_predictions.py',
    'model_comparison.py'
]

for script in scripts:
    print(f"\nRunning {script}...\n{'-' * 50}")
    # Define log file for each script
    log_file = os.path.join(logs_dir, f"{os.path.splitext(script)[0]}.log")
    
    # Run the script and capture output
    with open(log_file, 'w', encoding='utf-8') as log:
        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',  # Ensure UTF-8 encoding
            errors='replace',   # Replace undecodable characters
            env=dict(os.environ, PYTHONIOENCODING='utf-8')  # Set environment variable
        )

        # Stream the output to both console and log file
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)  # Write to console
            sys.stdout.flush()
            log.write(line)  # Write to log file

        process.stdout.close()
        return_code = process.wait()  # Wait for the process to finish

    # Check if the script exited with an error code
    if return_code != 0:
        print(f"{script} exited with an error code {return_code}. Stopping execution.\n{'-' * 50}")
        break
    print(f"Finished running {script}. Output saved to '{log_file}'.\n{'-' * 50}")
