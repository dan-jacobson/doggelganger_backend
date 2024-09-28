import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, pairwise_distances
from sklearn.linear_model import LinearRegression
import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

from doggelganger.utils import get_embedding, load_model as load_embedding_model
from doggelganger.models import LinearRegressionModel#, XGBoostModel, ResNetModel

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def make_training_data(data_dir):
    """
    Generate training data (X and y) from human and animal image pairs.

    This function processes image pairs from the specified data directory,
    generating embeddings for both human and animal images using a pre-trained model.

    Args:
        data_dir (str): Path to the directory containing 'human' and 'animal' subdirectories with image pairs.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - X (numpy.ndarray): Array of human image embeddings.
            - y (numpy.ndarray): Array of corresponding animal image embeddings.

    Raises:
        FileNotFoundError: If the specified data directory or required subdirectories do not exist.
        RuntimeError: If the model fails to load or generate embeddings.

    Note:
        - Image pairs should have the same filename in both 'human' and 'animal' subdirectories.
        - Skips image pairs where embedding generation fails for either the human or animal image.
    """
    pipe = load_embedding_model()
    X = []
    y = []

    human_dir = Path(data_dir) / "human"
    animal_dir = Path(data_dir) / "animal"

    for filename in tqdm(os.listdir(human_dir), desc="Processing image pairs"):
        human_path = human_dir / filename
        animal_path = animal_dir / filename

        if human_path.exists() and animal_path.exists():
            human_embedding = get_embedding(human_path, pipe)
            animal_embedding = get_embedding(animal_path, pipe)

            if human_embedding is not None and animal_embedding is not None:
                X.append(human_embedding)
                y.append(animal_embedding)
            else:
                logger.debug(f"Skipping {filename} due to embedding generation failure")

    logger.info(f"Processed {len(X)} valid image pairs")
    return np.array(X), np.array(y)


def train_model(model_class, X, y):
    """Train the specified model using the provided data.

    Args:
        model_class (BaseModel): The model class to use for training.
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target values.

    Returns:
        BaseModel: Trained model instance.

    Raises:
        ValueError: If no matching embeddings are found between human and animal datasets.
    """
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError(
            "No matching embeddings found between human and animal datasets"
        )

    model = model_class()
    model.fit(X, y)
    return model


def print_model_stats(model, X_train, y_train, X_test, y_test):
    """Print statistics about the trained model and return them.

    Args:
        model (LinearRegression): Trained LinearRegression model.
        X_train (numpy.ndarray): Training input features (human embeddings).
        y_train (numpy.ndarray): Training target values (animal embeddings).
        X_test (numpy.ndarray): Test input features (human embeddings).
        y_test (numpy.ndarray): Test target values (animal embeddings).

    Returns:
        dict: A dictionary containing the computed statistics.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    logger.debug(
        "\nModel Statistics:\n"
        f"  Number of training samples: {X_train.shape[0]}\n"
        f"  Number of test samples: {X_test.shape[0]}\n"
        f"  Training Mean Squared Error: {train_mse:.4f}\n"
        f"  Training R-squared Score: {train_r2:.4f}\n"
        f"  Test Mean Squared Error: {test_mse:.4f}\n"
        f"  Test R-squared Score: {test_r2:.4f}"
    )
    if isinstance(model.model, LinearRegression):
        logger.debug(
            f"  Coefficient shape: {model.model.coef_.shape}\n"
            f"  Intercept shape: {model.model.intercept_.shape}"
        )

    # Accuracy check using cosine similarity
    all_y = np.vstack((y_train, y_test))
    preds = model.predict(X_test)

    # Calculate cosine similarities
    cosine_similarities = 1 - pairwise_distances(preds, all_y, metric="cosine")

    # Calculate top-k accuracies
    n_test = len(y_test)
    correct_indices = np.arange(len(y_train), len(y_train) + n_test)

    top1_accuracy = np.mean(np.argmax(cosine_similarities, axis=1) == correct_indices)
    top3_accuracy = np.mean(
        [
            np.isin(correct_indices[i], indices[:3]).any()
            for i, indices in enumerate(np.argsort(-cosine_similarities, axis=1))
        ]
    )
    top10_accuracy = np.mean(
        [
            np.isin(correct_indices[i], indices[:10]).any()
            for i, indices in enumerate(np.argsort(-cosine_similarities, axis=1))
        ]
    )

    logger.debug(
        "\nAccuracy using Cosine Similarity:\n"
        f"Top-1 Accuracy: {top1_accuracy:.4f}\n"
        f"Top-3 Accuracy: {top3_accuracy:.4f}\n"
        f"Top-10 Accuracy: {top10_accuracy:.4f}"
    )

    return {
        "train_mse": train_mse,
        "train_r2": train_r2,
        "test_mse": test_mse,
        "test_r2": test_r2,
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "top10_accuracy": top10_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train the alignment model for Doggelganger"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for k-fold split (default: 1337)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "xgboost", "resnet"],
        default="linear",
        help="Model type to use (default: linear)",
    )
    parser.add_argument(
        "--save",
        type=bool or str,
        default=False,
        help="Saves model to /model/path (default: false)",
    )
    args = parser.parse_args()

    model_classes = {
        "linear": LinearRegressionModel,
        "xgboost": XGBoostModel,
        "resnet": ResNetModel
    }

    model_class = model_classes[args.model]

    try:
        # Load training data from /data/train
        X, y = make_training_data("data/train")

        # Initialize KFold
        n_splits = 8
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

        best_model = None
        best_score = float("-inf")

        # Lists to store statistics for each fold
        train_mse_list, train_r2_list = [], []
        test_mse_list, test_r2_list = [], []
        top1_acc_list, top3_acc_list, top10_acc_list = [], [], []

        for fold, (train_index, test_index) in tqdm(
            enumerate(kf.split(X), 1), total=n_splits, desc="Processing folds"
        ):
            logger.debug(f"\nFold {fold}/{n_splits}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the model
            model = train_model(model_class, X_train, y_train)

            # Print model statistics and get the values
            stats = print_model_stats(model, X_train, y_train, X_test, y_test)

            # Append statistics to lists
            train_mse_list.append(stats["train_mse"])
            train_r2_list.append(stats["train_r2"])
            test_mse_list.append(stats["test_mse"])
            test_r2_list.append(stats["test_r2"])
            top1_acc_list.append(stats["top1_accuracy"])
            top3_acc_list.append(stats["top3_accuracy"])
            top10_acc_list.append(stats["top10_accuracy"])

            # Save the best model based on test R-squared score
            if stats["test_r2"] > best_score:
                best_score = stats["test_r2"]
                best_model = model
                logger.info(f"New best model found (R-squared: {best_score:.4f})")

        # Print average statistics across all folds
        logger.info(
            "\nAverage Statistics Across All Folds:\n"
            f"  Training MSE: {np.mean(train_mse_list):.4f} (±{np.std(train_mse_list):.4f})\n"
            f"  Training R-squared: {np.mean(train_r2_list):.4f} (±{np.std(train_r2_list):.4f})\n"
            f"  Test MSE: {np.mean(test_mse_list):.4f} (±{np.std(test_mse_list):.4f})\n"
            f"  Test R-squared: {np.mean(test_r2_list):.4f} (±{np.std(test_r2_list):.4f})\n"
            f"  Top-1 Accuracy: {np.mean(top1_acc_list):.4f} (±{np.std(top1_acc_list):.4f})\n"
            f"  Top-3 Accuracy: {np.mean(top3_acc_list):.4f} (±{np.std(top3_acc_list):.4f})\n"
            f"  Top-10 Accuracy: {np.mean(top10_acc_list):.4f} (±{np.std(top10_acc_list):.4f})"
        )

        # Save the best alignment model
        if args.save:
            model_path = (
                args.save
                if isinstance(args.save, str)
                else f"weights/alignment_model_{args.model}.json"
            )
            best_model.save(model_path)

            logger.info(
                f"\nBest alignment model trained and saved. Used {len(X)} image pairs."
            )
            logger.info(f"Best model R-squared score: {best_score:.4f}")
            logger.info("This model now aligns human embeddings to animal embeddings.")
    except Exception as e:
        logger.error(f"An error occurred during the training process: {str(e)}")


if __name__ == "__main__":
    main()
