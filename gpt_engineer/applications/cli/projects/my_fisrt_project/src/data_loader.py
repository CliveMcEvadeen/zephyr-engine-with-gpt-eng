import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and preprocesses the data.

    Args:
        data_path (str): Path to the data file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and validation dataframes.
    """
    try:
        # Load data from CSV
        data = pd.read_csv(data_path)

        # Split into features and target
        X = data.drop("target_variable", axis=1)  # Replace "target_variable" with the actual target column name
        y = data["target_variable"]

        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        return (X_train, y_train), (X_val, y_val)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise