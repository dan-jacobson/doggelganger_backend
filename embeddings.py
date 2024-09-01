import os
import torch
import json
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from vecs import Client


# path to folder of dog images
dogs_folder = "example_dog_images"
image_types = (".png", ".jpg", ".jpeg", ".gif")
embeddings_cache = "dogs_embeddings.json"


def get_embedding(image_path, model, processor, device):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        return outputs.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


def calculate_dog_embeddings(folder_path):
    embeddings = {}
    image_files = [
        f for f in os.listdir(folder_path) if f.lower().endswith(image_types)
    ]

    for image_file in tqdm(image_files, desc="Calculating dog embeddings"):
        image_path = os.path.join(folder_path, image_file)
        embedding = get_embedding(
            image_path, model=model, processor=processor, device=device
        )
        embeddings[image_file] = embedding

    return embeddings


if __name__ == "__main__":
    # Load model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    print(f"Using device: {device}")

    cache_path = os.path.join(dogs_folder, embeddings_cache)
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            dog_embeddings = json.load(f)
            # Convert loaded lists back to numpy arrays
            dog_embeddings = {k: np.array(v) for k, v in dog_embeddings.items()}
            print(f"Loaded cached embeddings found at: {cache_path}")
    else:
        dog_embeddings = calculate_dog_embeddings(dogs_folder)
        # Convert numpy arrays to lists for JSON serialization
        json_embeddings = {k: v.tolist() for k, v in dog_embeddings.items()}
        with open(cache_path, "w") as f:
            json.dump(json_embeddings, f)
        print(f"Calculated embeddings for {len(dog_embeddings)} images.")
        print(f"Saved to: {cache_path}")
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from vecs import Client
from embeddings import get_embedding
from train import load_model, align_embedding
from transformers import CLIPProcessor, CLIPModel

# Supabase configuration
SUPABASE_URL = "YOUR_SUPABASE_URL"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"

def load_alignment_model(model_path):
    with open(model_path, 'r') as f:
        model_params = json.load(f)
    return np.array(model_params['coef']), np.array(model_params['intercept'])

def process_images(data_dir, metadata_path, alignment_model_path):
    # Load CLIP model
    model, processor, device = load_model()

    # Load alignment model
    coef, intercept = load_alignment_model(alignment_model_path)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Initialize Supabase client
    client = Client(SUPABASE_URL, SUPABASE_KEY)
    collection = client.get_or_create_collection("dog_embeddings")

    # Process images
    for filename in tqdm(os.listdir(data_dir), desc="Processing images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(data_dir, filename)
            
            # Get embedding
            embedding = get_embedding(image_path, model=model, processor=processor, device=device)
            
            if embedding is not None:
                # Align embedding
                aligned_embedding = align_embedding(embedding, coef, intercept)
                
                # Find corresponding metadata
                dog_metadata = next((item for item in metadata if item['image_file'] == filename), None)
                
                if dog_metadata:
                    # Prepare data for Supabase
                    vector_data = {
                        "id": dog_metadata['id'],
                        "vector": aligned_embedding.tolist(),
                        "metadata": dog_metadata
                    }
                    
                    # Push to Supabase
                    collection.upsert(vector_data)
                else:
                    print(f"No metadata found for {filename}")

def main():
    data_dir = "./data/petfinder"
    metadata_path = "./data/petfinder/dog_metadata.json"
    alignment_model_path = "./alignment_model.json"
    
    process_images(data_dir, metadata_path, alignment_model_path)
    print("Embeddings processed and pushed to Supabase.")

if __name__ == "__main__":
    main()
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from vecs import Client
from embeddings import get_embedding
from train import load_model, align_embedding
from transformers import CLIPProcessor, CLIPModel

# Supabase configuration
SUPABASE_URL = "YOUR_SUPABASE_URL"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"

def load_alignment_model(model_path):
    with open(model_path, 'r') as f:
        model_params = json.load(f)
    return np.array(model_params['coef']), np.array(model_params['intercept'])

def process_images(data_dir, metadata_path, alignment_model_path):
    # Load CLIP model
    model, processor, device = load_model()

    # Load alignment model
    coef, intercept = load_alignment_model(alignment_model_path)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Initialize Supabase client
    client = Client(SUPABASE_URL, SUPABASE_KEY)
    collection = client.get_or_create_collection("dog_embeddings")

    # Process images
    for filename in tqdm(os.listdir(data_dir), desc="Processing images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(data_dir, filename)
            
            # Get embedding
            embedding = get_embedding(image_path, model=model, processor=processor, device=device)
            
            if embedding is not None:
                # Align embedding
                aligned_embedding = align_embedding(embedding, coef, intercept)
                
                # Find corresponding metadata
                dog_metadata = next((item for item in metadata if item['image_file'] == filename), None)
                
                if dog_metadata:
                    # Prepare data for Supabase
                    vector_data = {
                        "id": dog_metadata['id'],
                        "vector": aligned_embedding.tolist(),
                        "metadata": dog_metadata
                    }
                    
                    # Push to Supabase
                    collection.upsert(vector_data)
                else:
                    print(f"No metadata found for {filename}")

def main():
    data_dir = "./data/petfinder"
    metadata_path = "./data/petfinder/dog_metadata.json"
    alignment_model_path = "./alignment_model.json"
    
    process_images(data_dir, metadata_path, alignment_model_path)
    print("Embeddings processed and pushed to Supabase.")

if __name__ == "__main__":
    main()
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from vecs import Client
from embeddings import get_embedding
from train import load_model, align_embedding
from transformers import CLIPProcessor, CLIPModel

# Supabase configuration
SUPABASE_URL = "YOUR_SUPABASE_URL"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"

def load_alignment_model(model_path):
    with open(model_path, 'r') as f:
        model_params = json.load(f)
    return np.array(model_params['coef']), np.array(model_params['intercept'])

def process_images(data_dir, metadata_path, alignment_model_path):
    # Load CLIP model
    model, processor, device = load_model()

    # Load alignment model
    coef, intercept = load_alignment_model(alignment_model_path)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Initialize Supabase client
    client = Client(SUPABASE_URL, SUPABASE_KEY)
    collection = client.get_or_create_collection("dog_embeddings")

    # Process images
    for filename in tqdm(os.listdir(data_dir), desc="Processing images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(data_dir, filename)
            
            # Get embedding
            embedding = get_embedding(image_path, model=model, processor=processor, device=device)
            
            if embedding is not None:
                # Align embedding
                aligned_embedding = align_embedding(embedding, coef, intercept)
                
                # Find corresponding metadata
                dog_metadata = next((item for item in metadata if item['image_file'] == filename), None)
                
                if dog_metadata:
                    # Prepare data for Supabase
                    vector_data = {
                        "id": dog_metadata['id'],
                        "vector": aligned_embedding.tolist(),
                        "metadata": dog_metadata
                    }
                    
                    # Push to Supabase
                    collection.upsert(vector_data)
                else:
                    print(f"No metadata found for {filename}")

def main():
    data_dir = "./data/petfinder"
    metadata_path = "./data/petfinder/dog_metadata.json"
    alignment_model_path = "./alignment_model.json"
    
    process_images(data_dir, metadata_path, alignment_model_path)
    print("Embeddings processed and pushed to Supabase.")

if __name__ == "__main__":
    main()
