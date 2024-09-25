import os
from pathlib import Path
import json
import logging
from tqdm import tqdm
import vecs
import hashlib
from dotenv import load_dotenv

from doggelganger.utils import load_model, get_embedding


load_dotenv()
DB_CONNECTION = os.getenv("SUPABASE_DB")

def generate_id(metadata):
    # Create a unique ID based on the dog's name and breed
    id_string = f"{metadata['adoption_link']}"
    return hashlib.md5(id_string.encode()).hexdigest()


def process_dogs(data_dir, metadata_path):
    pipe = load_model()

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
            dimension=pipe.model.config.hidden_size,
        )

        # Process dogs
        for dog in tqdm(metadata, desc="Processing dogs"):
            total_processed += 1

            try:
                # Determine image source
                if "local_image" in dog and dog["local_image"]:
                    image_path = Path(data_dir, dog["local_image"])
                    embedding = get_embedding(image_path, pipe)
                else:
                    embedding = get_embedding(dog["image_url"], pipe)

                if embedding is not None:
                    # Generate ID
                    dog_id = generate_id(dog)

                    # Prepare data for Supabase
                    record = (dog_id, embedding, dog)

                    # Push to Supabase
                    dogs.upsert([record])
                    successfully_pushed += 1
                else:
                    logging.error(f"Failed to get embedding for dog: {dog['name']}")
            except Exception as e:
                logging.error(f"Error processing dog {dog['name']}: {str(e)}")

        dogs.create_index()

    except Exception as e:
        logging.error(f"Error initializing Supabase client or collection: {str(e)}")

    logging.info(f"Total dogs processed: {total_processed}")
    logging.info(f"Successfully pushed to Supabase: {successfully_pushed}")


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    data_dir = "./data/petfinder"
    metadata_path = "./data/petfinder/dog_metadata.json"

    process_dogs(data_dir, metadata_path)
    logging.info("Embeddings processing completed.")


if __name__ == "__main__":
    main()
