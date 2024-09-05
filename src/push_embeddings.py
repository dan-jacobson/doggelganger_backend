import os
import json
import logging
import numpy as np
from tqdm import tqdm
import vecs
import hashlib
from dotenv import load_dotenv

from src.utils import load_model, get_embedding
from src.train import align_embedding

load_dotenv()

# Supabase configuration
DB_CONNECTION = os.getenv("SUPABASE_DB")


def load_alignment_model(model_path):
    with open(model_path, "r") as f:
        model_params = json.load(f)
    return np.array(model_params["coef"]), np.array(model_params["intercept"])


def generate_id(metadata):
    # Create a unique ID based on the dog's name and breed
    id_string = f"{metadata['adoption_link']}"
    return hashlib.md5(id_string.encode()).hexdigest()


def process_images(data_dir, metadata_path, alignment_model_path):
    # Load our model, constrolled by the env variable HUGGINGFACE_MODEL
    model, processor, device = load_model()

    # Load alignment model
    coef, intercept = load_alignment_model(alignment_model_path)

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Initialize counters
    total_processed = 0
    successfully_pushed = 0

    try:
        # Initialize Supabase client
        vx = vecs.create_client(DB_CONNECTION)
        dogs = vx.get_or_create_collection(
            name="dog_embeddings",
            dimension=model.config.hidden_size,
        )

        # Process images
        for filename in tqdm(os.listdir(data_dir), desc="Processing images"):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                image_path = os.path.join(data_dir, filename)
                total_processed += 1

                try:
                    # Get embedding
                    embedding = get_embedding(
                        image_path, model=model, processor=processor, device=device
                    )

                    if embedding is not None:
                        # Align embedding
                        aligned_embedding = align_embedding(embedding, coef, intercept)

                        # Find corresponding metadata
                        dog_metadata = next(
                            (
                                item
                                for item in metadata
                                if item["local_image"] == filename
                            ),
                            None,
                        )

                        if dog_metadata:
                            # Generate ID
                            dog_id = generate_id(dog_metadata)

                            # Prepare data for Supabase
                            record = (dog_id, aligned_embedding.tolist(), dog_metadata)

                            # Push to Supabase
                            dogs.upsert([record])
                            successfully_pushed += 1
                        else:
                            logging.warning(f"No metadata found for {filename}")
                    else:
                        logging.error(f"Failed to get embedding for {filename}")
                except Exception as e:
                    logging.error(f"Error processing {filename}: {str(e)}")
        dogs.create_index()

    except Exception as e:
        logging.error(f"Error initializing Supabase client or collection: {str(e)}")

    logging.info(f"Total images processed: {total_processed}")
    logging.info(f"Successfully pushed to Supabase: {successfully_pushed}")


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    data_dir = "./data/petfinder"
    metadata_path = "./data/petfinder/dog_metadata.json"
    alignment_model_path = "./alignment_model.json"

    process_images(data_dir, metadata_path, alignment_model_path)
    logging.info("Embeddings processing completed.")


if __name__ == "__main__":
    main()
