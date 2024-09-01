import json
import numpy as np
from scipy.spatial.distance import cosine

from embeddings import get_embedding

cache_path = "example_dog_images/dogs_embeddings.json"
alignment_model_path = "alignment_model.json"

def load_alignment_model(model_path):
    with open(model_path, 'r') as f:
        model_params = json.load(f)
    return np.array(model_params['coef']), np.array(model_params['intercept'])

def align_embedding(embedding, coef, intercept):
    return np.dot(embedding, coef.T) + intercept

def find_doggleganger(selfie_path, dog_embeddings, coef, intercept):
    selfie_embedding = get_embedding(selfie_path)
    aligned_selfie_embedding = align_embedding(selfie_embedding, coef, intercept)

    similarities = {}
    for dog_file, dog_embedding in dog_embeddings.items():
        similarity = 1 - cosine(aligned_selfie_embedding, dog_embedding)
        similarities[dog_file] = similarity

    best_match = max(similarities, key=similarities.get)
    return best_match, similarities[best_match]

# Usage
if __name__ == "__main__":
    selfie_path = "path_to_selfie.jpg"

    # Load dog embeddings
    with open(cache_path, 'r') as f:
        dog_embeddings = json.load(f)
        dog_embeddings = {k: np.array(v) for k, v in dog_embeddings.items()}

    # Load alignment model
    coef, intercept = load_alignment_model(alignment_model_path)

    best_match, similarity = find_doggleganger(selfie_path, dog_embeddings, coef, intercept)
    print(f"Your doggleganger is {best_match} with similarity {similarity:.4f}")
