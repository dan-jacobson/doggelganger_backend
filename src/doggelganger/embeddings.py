import argparse
import json
import logging
import os
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import jsonlines
import nest_asyncio
import requests
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from doggelganger.utils import load_model

nest_asyncio.apply()

load_dotenv()
DB_CONNECTION = os.getenv("SUPABASE_DB")
DOG_EMBEDDINGS_TABLE = os.getenv("DOG_EMBEDDINGS_TABLE")

# class DogDataset(Dataset):
#     def __init__(
#         self,
#         jsonl_path: list | str | Path,
#         transform: Optional[callable] = None,
#         field_to_embed: str = "primary_photo_cropped"
#     ):
#         self.data = []
#         if type(jsonl_path) is list:
#             self.data = jsonl_path
#         else:
#             self.jsonl_path = Path(jsonl_path)
#             with jsonlines.open(self.jsonl_path) as reader:
#                 self.data = list(reader)
#         self.transform = transform
#         self.field_to_embed = field_to_embed

#     async def _get_image(self, url, session):
#         try:
#             async with session.get(url) as response:
#                 if response.status == 200:
#                     data = await response.read()
#                     return Image.open(BytesIO(data))
#                 return None
#         except Exception as e:
#             print(f"Error downloading image {url}: {e}")
#             return None

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         url = item[self.field_to_embed]

#         # Create a new event loop for this thread if needed
#         try:
#             loop = asyncio.get_event_loop()
#         except RuntimeError:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)

#         # Create a new session for each request
#         async def fetch():
#             async with aiohttp.ClientSession() as session:
#                 return await self._get_image(url, session)

#         # Run async code in sync context
#         image = loop.run_until_complete(fetch())

#         if image is None:
#             return None

#         return {
#             "image": image,
#             "metadata": item
#         }


class DogDataset(Dataset):
    def __init__(
        self,
        jsonl_path: list | str | Path,
        transform: callable | None = None,
        field_to_embed: str = "primary_photo_cropped",
    ):
        self.data = []
        if type(jsonl_path) is list:
            self.data = jsonl_path
        else:
            self.jsonl_path = Path(jsonl_path)
            with jsonlines.open(self.jsonl_path) as reader:
                self.data = list(reader)
        self.transform = transform
        self.field_to_embed = field_to_embed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        url = item[self.field_to_embed]

        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))

            return {"image": image, "metadata": item}
        except Exception as e:
            print(f"Error downloading image {url}: {e}")
            return None


def collate_fn(batch):
    # filter out the images that failed to fetch
    batch = [b for b in batch if b is not None]

    if not batch:
        return None

    return {"images": [item["image"] for item in batch], "metadata": [item["metadata"] for item in batch]}


# def generate_id(metadata):
#     # Create a unique ID based on the dog's adoption link
#     id_string = f"{metadata['url']}"
#     return hashlib.md5(id_string.encode()).hexdigest()


# def process_dog(dog, pipe):
#     try:
#         # Get embedding from primary photo
#         embedding = get_embedding(dog["primary_photo_cropped"], pipe)

#         if embedding is not None:
#             # Generate ID
#             dog_id = generate_id(dog)
#             return dog_id, embedding, dog
#         else:
#             logging.error(f"Failed to get embedding for dog: {dog['name']}")
#             return None
#     except Exception as e:
#         logging.error(f"Error processing dog {dog['name']}: {str(e)}")
#         return None


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
    start_time = time.time()

    pipe = load_model(device="mps")

    # Load metadata
    metadata = load_metadata(metadata_path)
    logging.debug(f"Processing metadata file: {metadata_path}")
    logging.debug(f"Number of dogs found in file: {len(metadata)}")

    if smoke_test:
        metadata = metadata[:1000]
        logging.info(f"SMOKE TEST: Processing first {len(metadata)} dogs only")

    dataset = DogDataset(jsonl_path=metadata)

    batch_size = 16
    num_workers = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    # vx = vecs.create_client(DB_CONNECTION)

    # if drop_existing:
    #     vx.delete_collection(DOG_EMBEDDINGS_TABLE)
    #     logging.info(f"Existing '{DOG_EMBEDDINGS_TABLE}' collection dropped.")

    # dogs = vx.get_or_create_collection(
    #     name=DOG_EMBEDDINGS_TABLE,
    #     dimension=pipe.model.config.hidden_size,
    # )
    # # if we dropped and recreated the table, we need to make an index. we choose ivfflat
    # # because it's currently the best performing, and we can create the index *before* adding records.
    # if not dogs.index:
    #     dogs.create_index(method=vecs.IndexMethod.hnsw)

    embeddings = []
    # for out in tqdm(pipe(dataset,
    #                     batch_size=batch_size,
    #                     num_workers=8,
    #                 ), total=len(dataset)):

    #     embeddings.append(out)
    with tqdm(total=len(dataloader)) as pbar:
        embeddings = []
        for batch in dataloader:
            if not batch:
                pbar.update(1)
                continue

            emb = pipe(batch["images"], batch_size=batch_size)
            embeddings.append(emb)
            # for image in batch['images']:
            # embedding = get_embedding(image, pipe)
            # embeddings.append(embedding)

            pbar.update(1)
        # yield embeddings, batch['metadata']
    # Process dogs in parallel
    # process_dog_partial = partial(process_dog, pipe=pipe)

    # batch_size = 100
    # total_processed = 0
    # successfully_pushed = 0

    # if len(metadata) < N:
    #     N = len(metadata)
    #     logging.warning(
    #         f"--N flag set to {N}, but only {len(metadata)} dogs found in file. Processing {len(metadata)} dogs."
    #     )

    # if not N:
    #     N = len(metadata)

    # with ProcessPoolExecutor() as executor:
    #     progress_bar = tqdm(range(0, N, batch_size), desc="Processing dogs")
    #     for i in progress_bar:
    #         batch = metadata[i : i + batch_size]
    #         futures = [executor.submit(process_dog_partial, dog) for dog in batch]

    #         records = []
    #         for future in as_completed(futures):
    #             result = future.result()
    #             if result:
    #                 records.append(result)
    #                 successfully_pushed += 1
    #             total_processed += 1

    #         if records and not smoke_test:
    #             dogs.upsert(records)

    #         progress_bar.set_postfix({"success": successfully_pushed, "total": total_processed})

    end_time = time.time()
    duration = end_time - start_time

    logging.info("=== Benchmark Results ===")
    logging.info(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total duration: {duration:.2f} seconds")
    logging.info(f"Total dogs processed: {1000}")
    logging.info(f"Successfully processed: {len(embeddings)}")
    logging.info(f"Average time per dog: {duration / 1000:.2f} seconds")
    logging.info(f"Dogs per second: {1000 / duration:.2f}")

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
