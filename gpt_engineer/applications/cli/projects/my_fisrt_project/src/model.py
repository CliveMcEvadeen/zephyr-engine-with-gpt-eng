from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(X_train, y_train, epochs: int, batch_size: int, learning_rate: float):
    """Trains a logistic regression model.

    Args:
        X_train: Training data features.
        y_train: Training data labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: LogisticRegression, X_val, y_val) -> Dict:
    """Evaluates the trained model.

    Args:
        model (LogisticRegression): The trained logistic regression model.
        X_val: Validation data features.
        y_val: Validation data labels.

    Returns:
        Dict: A dictionary containing the evaluation metrics.
    """
    y_pred = model.predict(X_val)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
    }