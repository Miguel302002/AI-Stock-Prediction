# AI Stock Prediction Project

This project aims to predict stock price movements using machine learning models, specifically a Random Forest classifier and an LSTM neural network.

## Project Overview

The project consists of several Python scripts that perform data preprocessing, model training, evaluation, and visualization. The main goal is to predict whether the closing price of a stock will increase or decrease the next day.

## Getting Started

### Prerequisites

- Python 3.x
- Virtual Environment (optional but recommended)
- Required Python libraries (see `requirements.txt`)

### Setup Instructions

1. **Clone the Repository**

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**

   - On Windows:
        ```bash
        venv\Scripts\activate
        ```

   - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4. **Install Required Libraries**

    ```bash
    pip install -r requirements.txt
    ```

   **Note:** The `requirements.txt` file should contain all the necessary libraries, such as `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, etc.

5. **Prepare the Dataset**

   - Ensure you have the dataset file named `major-tech-stock-2019-2024.csv` in the project directory.
   - The dataset should contain historical stock data for major tech companies from 2019 to 2024.

## Project Files and Their Functions

### 1. `data_preprocessing.py`

- **Purpose:** Preprocesses the raw stock data.
- **What It Does:**
  - Loads the dataset.
  - Handles missing data by forward and backward filling.
  - Performs feature engineering to create technical indicators like moving averages and volatility.
  - Splits the data into training and testing sets.
  - Scales the features.
  - Saves the preprocessed data to `processed_data.csv`.

### 2. `random_forest.py`

- **Purpose:** Trains and evaluates a Random Forest classifier.
- **What It Does:**
  - Loads the preprocessed data.
  - Creates the target variable indicating if the next day's closing price will increase.
  - Splits the data into features and target for training and testing.
  - Performs hyperparameter tuning using `RandomizedSearchCV`.
  - Trains the Random Forest model with the best parameters.
  - Evaluates the model and saves the results.
  - Saves the trained model to `random_forest_model.pkl`.

### 3. `lstm_model.py`

- **Purpose:** Trains and evaluates an LSTM neural network.
- **What It Does:**
  - Loads the preprocessed data.
  - Creates sequences of data for time series prediction.
  - Handles class imbalance by calculating class weights.
  - Builds and trains the LSTM model.
  - Evaluates the model and saves the results.
  - Saves the trained model to `lstm_model.keras`.

### 4. `visualize_predictions.py`

- **Purpose:** Generates visualizations of the model predictions.
- **What It Does:**
  - Loads the test and predicted values for both models.
  - Creates plots comparing actual vs. predicted values.
  - Saves the plots as `lstm_actual_vs_predicted.png` and `rf_actual_vs_predicted.png`.

### 5. `model_comparison.py`

- **Purpose:** Compares the performance of both models.
- **What It Does:**
  - Loads the accuracy and classification reports of both models.
  - Prints out the results for easy comparison.

### 6. `run_all.py`

- **Purpose:** Runs all the above scripts sequentially.
- **What It Does:**
  - Executes each script one after the other.
  - Ensures that all steps are completed in the correct order.
  - Captures and displays the output of each script.

### 7. `evaluation.py`

- **Purpose:** Contains functions for model evaluation.
- **What It Does:**
  - Provides functions to generate confusion matrices and classification reports.
  - Saves evaluation results for further analysis.

## How to Run the Project

```bash
python run_all.py
```

### Run All Scripts Together

To execute the entire workflow with a single command:

This will run all the scripts in the correct order:

1. `data_preprocessing.py`
2. `random_forest.py`
3. `lstm_model.py`
4. `visualize_predictions.py`
5. `model_comparison.py`

### Run Scripts Individually

If you prefer to run the scripts one by one:

1. **Data Preprocessing**

    ```bash
    python data_preprocessing.py
    ```

2. **Random Forest Model**

    ```bash
    python random_forest.py
    ```

3. **LSTM Model**

    ```bash
    python lstm_model.py
    ```

4. **Visualization**

    ```bash
    python visualize_predictions.py
    ```

5. **Model Comparison**

    ```bash
    python model_comparison.py
    ```

**Note:** Ensure that you run `data_preprocessing.py` before any of the model scripts, as they rely on the preprocessed data.

## Additional Details

- **Output Files:**
  - Preprocessed data: `processed_data.csv`
  - Random Forest model: `random_forest_model.pkl`
  - LSTM model: `lstm_model.keras`
  - Evaluation results and plots are saved in the project directory.

- **Logging and Outputs:**
  - The scripts print progress and results to the console.
  - Any warnings or errors are displayed during execution.

- **Virtual Environment:**
  - Using a virtual environment is recommended to manage dependencies and avoid conflicts.
  - Remember to activate the virtual environment before running the scripts.

- **Dependencies:**
  - Ensure all required libraries are installed as per `requirements.txt`.
  - If you encounter any missing packages, you can install them using `pip install package_name`.

## Troubleshooting

- **Encoding Errors:**
  - If you encounter `UnicodeEncodeError`, ensure that the encoding is set to UTF-8.
  - This has been handled in the scripts, but if issues persist, check your system's default encoding.

- **Deprecation Warnings:**
  - The scripts have been updated to avoid deprecated functions.
  - If you see any warnings, make sure all packages are up to date.

- **Errors During Execution:**
  - If a script exits with an error, read the error message carefully.
  - Common issues include missing data files or package dependencies.
