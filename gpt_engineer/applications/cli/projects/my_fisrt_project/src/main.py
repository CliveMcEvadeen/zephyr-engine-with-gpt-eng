import argparse
import logging
from typing import Dict

import mlflow

from data_loader import load_data
from model import train_model, evaluate_model

# Configure logging
logging.basicConfig(level=logging.INFO)

def parse_arguments() -> Dict:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate a machine learning model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--model_name", type=str, default="my_model", help="Name of the model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    return vars(parser.parse_args())

def main():
    """Main function to train and evaluate the model."""
    args = parse_arguments()

    # Start MLflow run
    with mlflow.start_run(run_name=args["model_name"]):
        # Log parameters
        mlflow.log_params(args)

        # Load data
        logging.info("Loading data...")
        train_data, val_data = load_data(args["data_path"])

        # Train model
        logging.info("Training model...")
        model = train_model(train_data, args["epochs"], args["batch_size"], args["learning_rate"])

        # Evaluate model
        logging.info("Evaluating model...")
        metrics = evaluate_model(model, val_data)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()