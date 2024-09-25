import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
import os
from pathlib import Path
from tqdm import tqdm

from doggelganger.utils import load_model, get_embedding


def make_embeddings(data_dir):
    pipe = load_model()
    human_embeddings = {}
    animal_embeddings = {}

    human_dir = Path(data_dir) / "human"
    animal_dir = Path(data_dir) / "animal"

    for filename in tqdm(os.listdir(human_dir), desc="Processing image pairs"):
        human_path = human_dir / filename
        animal_path = animal_dir / filename

        if human_path.exists() and animal_path.exists():
            human_embedding = get_embedding(human_path, pipe)
            animal_embedding = get_embedding(animal_path, pipe)

            if human_embedding is not None and animal_embedding is not None:
                human_embeddings[filename] = human_embedding
                animal_embeddings[filename] = animal_embedding
            else:
                print(f"Skipping {filename} due to embedding generation failure")

    print(f"Processed {len(human_embeddings)} valid image pairs")
    return human_embeddings, animal_embeddings


def align_human_to_animal_embeddings(human_embeddings, animal_embeddings):
    """Align human embeddings to animal embeddings using a linear transformation.

    Args:
        human_embeddings (dict): Dictionary of human embeddings.
        animal_embeddings (dict): Dictionary of animal embeddings.

    Returns:
        tuple: A tuple containing:
            - model (LinearRegression): Trained LinearRegression model.
            - X (numpy.ndarray): Human embeddings (input).
            - y (numpy.ndarray): Corresponding animal embeddings (target).

    Raises:
        ValueError: If no matching embeddings are found between human and animal datasets.
    """
    X = []  # human embeddings (input)
    y = []  # corresponding animal embeddings (target)

    for filename in human_embeddings.keys():
        if filename in animal_embeddings.keys():
            X.append(human_embeddings[filename])
            y.append(animal_embeddings[filename])

    if not X or not y:
        raise ValueError(
            "No matching embeddings found between human and animal datasets"
        )

    X = np.array(X)
    y = np.array(y)

    model = LinearRegression()
    model.fit(X, y)
    return model, X, y


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


def print_model_stats(model, X, y):
    """Print statistics about the trained model.

    Args:
        model (LinearRegression): Trained LinearRegression model.
        X (numpy.ndarray): Input features (human embeddings).
        y (numpy.ndarray): Target values (animal embeddings).
    """
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("\nModel Training Statistics:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Coefficient shape: {model.coef_.shape}")
    print(f"Intercept shape: {model.intercept_.shape}")


def main():
    try:
        # Load embeddings from /data/train
        human_embeddings, animal_embeddings = make_embeddings("data/train")

        # Align human embeddings to animal embeddings
        alignment_model, X, y = align_human_to_animal_embeddings(
            human_embeddings, animal_embeddings
        )

        # Print model statistics
        print_model_stats(alignment_model, X, y)

        # Save the alignment model
        model_params = {
            "coef": alignment_model.coef_.tolist(),
            "intercept": alignment_model.intercept_.tolist(),
        }
        with open("weights/alignment_model.json", "w") as f:
            json.dump(model_params, f)

        print(f"\nAlignment model trained and saved. Used {len(X)} image pairs.")
        print("This model now aligns human embeddings to animal embeddings.")
    except Exception as e:
        print(f"An error occurred during the training process: {str(e)}")


if __name__ == "__main__":
    main()
