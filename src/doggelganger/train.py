import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, pairwise_distances
import json
import os
from pathlib import Path
from tqdm import tqdm

from doggelganger.utils import load_model, get_embedding


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
    pipe = load_model()
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
                print(f"Skipping {filename} due to embedding generation failure")

    print(f"Processed {len(X)} valid image pairs")
    return np.array(X), np.array(y)


def align_human_to_animal_embeddings(X, y):
    """Align human embeddings to animal embeddings using a linear transformation.

    Args:
        X (numpy.ndarray): Human embeddings (input).
        y (numpy.ndarray): Corresponding animal embeddings (target).

    Returns:
        LinearRegression: Trained LinearRegression model.

    Raises:
        ValueError: If no matching embeddings are found between human and animal datasets.
    """
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError(
            "No matching embeddings found between human and animal datasets"
        )

    model = LinearRegression()
    model.fit(X, y)
    return model


def align_embedding(embedding, coef, intercept):
    """Align a single human embedding to the animal embedding space using the trained linear transformation.

    Args:
        embedding (numpy.ndarray): Numpy array of the human embedding to align.
        coef (numpy.ndarray): Coefficient matrix from the trained LinearRegression model.
        intercept (numpy.ndarray): Intercept vector from the trained LinearRegression model.

    Returns:
        numpy.ndarray: Aligned embedding in the animal embedding space.
    """
    return np.dot(embedding, coef.T) + intercept


def print_model_stats(model, X_train, y_train, X_test, y_test):
    """Print statistics about the trained model.

    Args:
        model (LinearRegression): Trained LinearRegression model.
        X_train (numpy.ndarray): Training input features (human embeddings).
        y_train (numpy.ndarray): Training target values (animal embeddings).
        X_test (numpy.ndarray): Test input features (human embeddings).
        y_test (numpy.ndarray): Test target values (animal embeddings).
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nModel Statistics:")
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of test samples: {X_test.shape[0]}")
    print(f"Training Mean Squared Error: {train_mse:.4f}")
    print(f"Training R-squared Score: {train_r2:.4f}")
    print(f"Test Mean Squared Error: {test_mse:.4f}")
    print(f"Test R-squared Score: {test_r2:.4f}")
    print(f"Coefficient shape: {model.coef_.shape}")
    print(f"Intercept shape: {model.intercept_.shape}")

    # Accuracy check using cosine similarity
    all_y = np.vstack((y_train, y_test))
    y_test_aligned = model.predict(X_test)
    
    # Calculate cosine similarities
    cosine_similarities = 1 - pairwise_distances(y_test_aligned, all_y, metric='cosine')
    
    # Calculate top-k accuracies
    top1_accuracy = np.mean(np.argmax(cosine_similarities, axis=1) == np.arange(len(y_test)))
    top3_accuracy = np.mean([np.isin(np.arange(len(y_test)), indices[:3]).any() 
                             for indices in np.argsort(-cosine_similarities, axis=1)])
    top10_accuracy = np.mean([np.isin(np.arange(len(y_test)), indices[:10]).any() 
                              for indices in np.argsort(-cosine_similarities, axis=1)])
    
    print("\nAccuracy using Cosine Similarity:")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
    print(f"Top-10 Accuracy: {top10_accuracy:.4f}")


def main():
    try:
        # Load training data from /data/train
        X, y = make_training_data("data/train")

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

        # Align human embeddings to animal embeddings
        alignment_model = align_human_to_animal_embeddings(X_train, y_train)

        # Print model statistics
        print_model_stats(alignment_model, X_train, y_train, X_test, y_test)

        # Save the alignment model
        model_params = {
            "coef": alignment_model.coef_.tolist(),
            "intercept": alignment_model.intercept_.tolist(),
        }
        with open("weights/alignment_model.json", "w") as f:
            json.dump(model_params, f)

        print(f"\nAlignment model trained and saved. Used {X.shape[0]} image pairs.")
        print("This model now aligns human embeddings to animal embeddings.")
    except Exception as e:
        print(f"An error occurred during the training process: {str(e)}")


if __name__ == "__main__":
    main()
