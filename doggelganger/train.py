import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from utils import load_model, get_embedding


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

    return human_embeddings, animal_embeddings


def align_animal_to_human_embeddings(human_embeddings, animal_embeddings):
    """
    Align animal embeddings to human embeddings using a linear transformation.

    :param human_embeddings: dict of human embeddings
    :param animal_embeddings: dict of animal embeddings
    :return: trained LinearRegression model, X (animal embeddings), y (human embeddings)
    """
    X = []  # animal embeddings (input)
    y = []  # corresponding human embeddings (target)

    for filename in human_embeddings.keys():
        if filename in animal_embeddings.keys():
            X.append(animal_embeddings[filename])
            y.append(human_embeddings[filename])

    X = np.array(X)
    y = np.array(y)

    model = LinearRegression()
    model.fit(X, y)
    return model, X, y


def align_embedding(embedding, coef, intercept):
    """
    Align a single embedding using the trained linear transformation.

    :param embedding: numpy array of the embedding to align
    :param coef: coefficient matrix from the trained LinearRegression model
    :param intercept: intercept vector from the trained LinearRegression model
    :return: aligned embedding
    """
    return np.dot(embedding, coef.T) + intercept


def print_model_stats(model, X, y):
    """
    Print statistics about the trained model.

    :param model: trained LinearRegression model
    :param X: input features (dog embeddings)
    :param y: target values (face embeddings)
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
    # Load embeddings from /data/train
    human_embeddings, animal_embeddings = make_embeddings("data/train")

    # Align animal embeddings to human embeddings
    alignment_model, X, y = align_animal_to_human_embeddings(
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


if __name__ == "__main__":
    main()
