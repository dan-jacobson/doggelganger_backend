import argparse
import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import jsonlines
import vecs
from dotenv import load_dotenv
from tqdm import tqdm

from doggelganger.utils import get_embedding, load_model

load_dotenv()
DB_CONNECTION = os.getenv("SUPABASE_DB")
DOG_EMBEDDINGS_TABLE = os.getenv("DOG_EMBEDDINGS_TABLE")


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
    if suffix == ".jsonl":
        with jsonlines.open(metadata_path) as reader:
            return list(reader)
    else:  # Assume .json for all other cases
        with open(metadata_path) as f:
            return json.load(f)


def process_dogs(metadata_path, drop_existing: bool = False, N: int | bool = False, smoke_test: bool = False):
    pipe = load_model(device='cpu')

    # Load metadata
    metadata = load_metadata(metadata_path)
    logging.debug(f"Processing metadata file: {metadata_path}")
    logging.debug(f"Number of dogs found in file: {len(metadata)}")

    if smoke_test:
        metadata = metadata[:10]
        logging.info(f"SMOKE TEST: Processing first {len(metadata)} dogs only")

    vx = vecs.create_client(DB_CONNECTION)

    if drop_existing:
        vx.delete_collection(DOG_EMBEDDINGS_TABLE)
        logging.info(f"Existing '{DOG_EMBEDDINGS_TABLE}' collection dropped.")

    dogs = vx.get_or_create_collection(
        name=DOG_EMBEDDINGS_TABLE,
        dimension=pipe.model.config.hidden_size,
    )
    # if we dropped and recreated the table, we need to make an index. we choose ivfflat
    # because it's currently the best performing, and we can create the index *before* adding records.
    if not dogs.index:
        dogs.create_index(method=vecs.IndexMethod.hnsw)

    # Process dogs in parallel
    process_dog_partial = partial(process_dog, pipe=pipe)

    batch_size = 100
    total_processed = 0
    successfully_pushed = 0

    if len(metadata) < N:
        N = len(metadata)
        logging.warning(
            f"--N flag set to {N}, but only {len(metadata)} dogs found in file. Processing {len(metadata)} dogs."
        )

    if not N:
        N = len(metadata)

    with ProcessPoolExecutor() as executor:
        progress_bar = tqdm(range(0, N, batch_size), desc="Processing dogs")
        for i in progress_bar:
            batch = metadata[i : i + batch_size]
            futures = [executor.submit(process_dog_partial, dog) for dog in batch]

            records = []
            for future in as_completed(futures):
                result = future.result()
                if result:
                    records.append(result)
                    successfully_pushed += 1
                total_processed += 1

            if records and not smoke_test:
                dogs.upsert(records)

            progress_bar.set_postfix({"success": successfully_pushed, "total": total_processed})

    if smoke_test:
        logging.info("SMOKE TEST RESULTS:")
        logging.info(f"Total dogs processed: {total_processed}")
        logging.info(f"Successfully generated embeddings: {successfully_pushed}")
        logging.info("No data was written to the database")
    else:
        logging.info(f"Total dogs processed: {total_processed}")
        logging.info(f"Successfully pushed to Supabase: {successfully_pushed}")


def main():
    parser = argparse.ArgumentParser(description="Process dog embeddings and upload to Supabase.")
    parser.add_argument("--file", required=True, help="Path to the dog metadata file (JSON or JSONL format)")
    parser.add_argument("--drop", action="store_true", help="Drop existing collection before processing")
    parser.add_argument("--N", default=False, help="Number of embeddings to process")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Process first 10 dogs only and print results without writing to DB"
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    process_dogs(args.file, drop_existing=args.drop, smoke_test=args.smoke_test, N=int(args.N))
    logging.info("Embeddings processing completed.")


if __name__ == "__main__":
    main()
