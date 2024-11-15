import os
from pathlib import Path
import json
import jsonlines
import logging
from tqdm import tqdm
import vecs
import hashlib
from dotenv import load_dotenv
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from doggelganger.utils import load_model, get_embedding


load_dotenv()
DB_CONNECTION = os.getenv("SUPABASE_DB")


def generate_id(metadata):
    # Create a unique ID based on the dog's adoption link
    id_string = f"{metadata['url']}"
    return hashlib.md5(id_string.encode()).hexdigest()


def process_dog(dog, pipe):
    try:
        # Get embedding from primary photo
        embedding = get_embedding(dog["primary_photo_cropped"], pipe)

        if embedding is not None:
            # Generate ID
            dog_id = generate_id(dog)
            return dog_id, embedding, dog
        else:
            logging.error(f"Failed to get embedding for dog: {dog['name']}")
            return None
    except Exception as e:
        logging.error(f"Error processing dog {dog['name']}: {str(e)}")
        return None


def load_metadata(metadata_path):
    """Load metadata from either JSON or JSONL file."""
    suffix = Path(metadata_path).suffix.lower()
    if suffix == '.jsonl':
        with jsonlines.open(metadata_path) as reader:
            return list(reader)
    else:  # Assume .json for all other cases
        with open(metadata_path, "r") as f:
            return json.load(f)

def process_dogs(metadata_path, drop_existing=False):
    pipe = load_model()

    # Load metadata
    metadata = load_metadata(metadata_path)

    # Initialize Supabase client
    vx = vecs.create_client(DB_CONNECTION)
    
    if drop_existing:
        vx.delete_collection("dog_embeddings")
        logging.info("Existing 'dog_embeddings' collection dropped.")

    dogs = vx.get_or_create_collection(
        name="dog_embeddings",
        dimension=pipe.model.config.hidden_size,
    )

    # Process dogs in parallel
    process_dog_partial = partial(process_dog, pipe=pipe)
    
    batch_size = 100
    total_processed = 0
    successfully_pushed = 0

    with ProcessPoolExecutor() as executor:
        for i in range(0, len(metadata), batch_size):
            batch = metadata[i:i+batch_size]
            futures = [executor.submit(process_dog_partial, dog) for dog in batch]
            
            records = []
            for future in as_completed(futures):
                result = future.result()
                if result:
                    records.append(result)
                    successfully_pushed += 1
                total_processed += 1

            if records:
                dogs.upsert(records)

            logging.info(f"Processed {total_processed} dogs, successfully pushed {successfully_pushed}")

    dogs.create_index()
    logging.info(f"Total dogs processed: {total_processed}")
    logging.info(f"Successfully pushed to Supabase: {successfully_pushed}")


def main():
    parser = argparse.ArgumentParser(description="Process dog embeddings and upload to Supabase.")
    parser.add_argument('--file', required=True, help='Path to the dog metadata file (JSON or JSONL format)')
    parser.add_argument('--drop', action='store_true', help='Drop existing collection before processing')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    process_dogs(args.file, drop_existing=args.drop)
    logging.info("Embeddings processing completed.")


if __name__ == "__main__":
    main()
