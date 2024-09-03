import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
import os
import torch
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL")


def load_model():
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    processor = AutoImageProcessor.from_pretrained(HUGGINGFACE_MODEL)
    model = AutoModel.from_pretrained(HUGGINGFACE_MODEL)
    torch.compile(model)

    model.to(device)

    return model, processor, device

def get_embedding(img, processor, model, device):
    try:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            raise ValueError("Input must be either a file path or a PIL Image object")
        
        inputs = processor(images=img, return_tensors="pt")
        inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.pooler_output.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def make_embeddings(data_dir):
    model, processor, device = load_model()
    selfie_embeddings = {}
    dog_embeddings = {}
    examples = []

    for filename in tqdm(os.listdir(data_dir), desc="Processing images"):
        if "H" in filename:
            selfie_path = os.path.join(data_dir, filename)
            dog_path = os.path.join(data_dir, filename.replace("H", "D"))
            selfie = Image.open(selfie_path).convert("RGB")
            dog = Image.open(dog_path).convert("RGB")

            if os.path.exists(dog_path):
                selfie_embedding = get_embedding(
                    selfie_path, processor=processor, model=model, device=device
                )
                dog_embedding = get_embedding(
                    selfie_path, processor=processor, model=model, device=device
                )

                if selfie_embedding is not None and dog_embedding is not None:
                    selfie_embeddings[filename] = selfie_embedding
                    dog_embeddings[filename.replace("H", "D")] = dog_embedding
                    examples.append((filename, filename.replace("H", "D")))

    return selfie_embeddings, dog_embeddings, examples


def align_dog_to_face_embeddings(face_embeddings, dog_embeddings, examples):
    """
    Align dog embeddings to face embeddings using a linear transformation.

    :param face_embeddings: dict of face (selfie) embeddings
    :param dog_embeddings: dict of dog embeddings
    :param examples: list of tuples (selfie_name, dog_name) for alignment
    :return: trained LinearRegression model, X (dog embeddings), y (face embeddings)
    """
    X = []  # dog embeddings (input)
    y = []  # corresponding face embeddings (target)

    for selfie_name, dog_name in examples:
        if selfie_name in face_embeddings and dog_name in dog_embeddings:
            X.append(dog_embeddings[dog_name])
            y.append(face_embeddings[selfie_name])

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
    # Load embeddings and examples from /data/train
    face_embeddings, dog_embeddings, examples = make_embeddings("./data/train")

    # Align dog embeddings to face embeddings
    alignment_model, X, y = align_dog_to_face_embeddings(face_embeddings, dog_embeddings, examples)

    # Print model statistics
    print_model_stats(alignment_model, X, y)

    # Save the alignment model
    model_params = {
        "coef": alignment_model.coef_.tolist(),
        "intercept": alignment_model.intercept_.tolist(),
    }
    with open("alignment_model.json", "w") as f:
        json.dump(model_params, f)

    print(f"\nAlignment model trained and saved. Used {len(examples)} image pairs.")


if __name__ == "__main__":
    main()
