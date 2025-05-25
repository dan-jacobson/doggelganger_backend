import argparse
import asyncio
import logging
import os
from dataclasses import asdict
from io import BytesIO
from queue import Queue
from typing import NamedTuple

import aiohttp
import jsonlines
import vecs
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Pipeline

from doggelganger.utils import Animal, load_model


# I would've used a dataclass, but it doesn't play nicely with `vecs`
class Record(NamedTuple):
    id: str
    embedding: list
    metadata: dict


load_dotenv()
DB_CONNECTION = os.getenv("SUPABASE_DB")
DOG_EMBEDDINGS_TABLE = os.getenv("DOG_EMBEDDINGS_TABLE")
BATCH_SIZE = 32


def load_metadata(metadata_path) -> list[Animal]:
    """Load metadata from either JSON or JSONL file."""
    with jsonlines.open(metadata_path) as reader:
        return [Animal(**data) for data in reader]


class AsyncDogDataset(Dataset):
    def __init__(
        self,
        metadata: list,
        field_to_embed: str = "primary_photo_cropped",
        batch_size: int = 32,
        queue_size: int = 1000,
        num_fetchers: int = 8,
        show_progress: bool = True,
    ):
        self.metadata = metadata
        self.field_to_embed = field_to_embed
        self.batch_size = batch_size
        self.image_queue = asyncio.Queue(maxsize=queue_size)
        self.processed_queue = Queue()
        self.num_fetchers = num_fetchers
        self.total_items = len(metadata)
        self.show_progess = show_progress

        # Useful for preventing bad behavior in CI and testing
        if self.show_progess:
            self.fetch_pbar = tqdm(total=self.total_items, desc="Fetching images", position=0)
            self.embed_pbar = tqdm(total=self.total_items, desc="Processing embeddings", position=1)
            print("\n")
        else:
            self.fetch_pbar = None
            self.embed_pbar = None

    async def fetch_image(self, session: aiohttp.ClientSession, item: Animal) -> tuple[dict, bytes | None]:
        url = getattr(item, self.field_to_embed)
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.read()
                    return item, data
                logging.warning(f"Failed to fetch {url}: Status {response.status}")
                return item, None
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            return item, None

    async def producer(self):
        async with aiohttp.ClientSession() as session:
            chunk_size = 1000
            for i in range(0, len(self.metadata), chunk_size):
                chunk = self.metadata[i : i + chunk_size]
                tasks = [self.fetch_image(session, item) for item in chunk]

                for task in asyncio.as_completed(tasks):
                    item, image_data = await task
                    if image_data:
                        try:
                            image = Image.open(BytesIO(image_data))
                            await self.image_queue.put((item, image))
                            if self.fetch_pbar:
                                self.fetch_pbar.update(1)
                        except Exception as e:
                            logging.error(f"Error processing image: {e}")

        # stop signal for producer: we've processed all the images
        await self.image_queue.put(None)
        if self.fetch_pbar:
            self.fetch_pbar.close()

    def generate_records(self, model: Pipeline, images, metadata: list[Animal]) -> list[Record]:
        embeddings = model(images, batch_size=len(images))

        # gotta un-nest the embeddings
        embeddings = [e[0] for e in embeddings]

        metadata = [asdict(m) for m in metadata]

        # pop out ids, make them strings for `vecs`
        ids = [str(m.pop("id")) for m in metadata]

        return [Record(id, e, m) for id, e, m in zip(ids, embeddings, metadata, strict=False)]

    async def consumer(self, model: Pipeline, db: vecs.Collection = None):
        current_batch_images = []
        current_batch_metadata = []

        async def handle_batch(images, metadata, db):
            records = self.generate_records(model, images, metadata)
            self.processed_queue.put(records)
            if db is not None:
                try:
                    db.upsert(records)
                except Exception as e:
                    logging.error(f"Failed to upsert records: {e}")
            if self.embed_pbar:
                self.embed_pbar.update(len(records))
            return [], []

        while True:
            try:
                item = await self.image_queue.get()

                # Stop signal, process whatever we have left
                if item is None:
                    if current_batch_images:
                        await handle_batch(current_batch_images, current_batch_metadata, db)
                    break

                metadata, image = item
                current_batch_images.append(image)
                current_batch_metadata.append(metadata)

                if len(current_batch_images) >= self.batch_size:
                    current_batch_images, current_batch_metadata = await handle_batch(
                        current_batch_images, current_batch_metadata, db
                    )

            except Exception as e:
                logging.error(f"Error in consumer: {e}")
                continue

        # stop signal for consumer
        self.processed_queue.put(None)
        if self.embed_pbar:
            self.embed_pbar.close()

    async def process_all(self, model: Pipeline, db: vecs.Collection = None):
        producer_task = asyncio.create_task(self.producer())
        consumer_task = asyncio.create_task(self.consumer(model, db))

        await asyncio.gather(producer_task, consumer_task)

        records = []

        while True:
            result = self.processed_queue.get()
            if result is None:
                break
            records += result

        return records

    def __call__(self, model: Pipeline, db: vecs.Collection = None):
        """Makes the dataset callable, trying to make this flow a bit more pythonic"""
        return asyncio.run(self.process_all(model, db))


def process_dogs(metadata_path, drop_existing: bool = False, N: int | bool = False, smoke_test: bool = False):
    pipe = load_model(device="mps")

    # Load metadata
    metadata = load_metadata(metadata_path)
    logging.debug(f"Processing metadata file: {metadata_path}")
    logging.debug(f"Number of dogs found in file: {len(metadata)}")

    if smoke_test:
        metadata = metadata[:1000]
        logging.info(f"SMOKE TEST: Processing first {len(metadata)} dogs only")
        # we're not going to touch the DB so we set to None
        dogs = None
    else:
        if N:
            metadata = metadata[:N]
            logging.debug(f"--N flag set, number of dogs to process: {N}")
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

    # I was tryna make this feel pythonic, and dataset(model) was as good as I could come up with
    dataset = AsyncDogDataset(metadata=metadata, batch_size=BATCH_SIZE)
    records = dataset(pipe, db=dogs)

    if smoke_test:
        logging.info("SMOKE TEST RESULTS:")
        logging.info(f"Total dogs processed: {len(metadata)}")
        logging.info(f"Successfully generated embeddings: {len(records)}")
        logging.info("No data was written to the database")
    else:
        logging.info(f"Total dogs processed: {len(metadata)}")
        logging.info(f"Successfully pushed to Supabase: {len(records)}")


def main():
    parser = argparse.ArgumentParser(description="Process dog embeddings and upload to Supabase.")
    parser.add_argument("--file", required=True, help="Path to the dog metadata file (JSON or JSONL format)")
    parser.add_argument("--drop", action="store_true", help="Drop existing collection before processing")
    parser.add_argument("--N", default=False, help="Number of embeddings to process")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Process first 10 dogs only and print results without writing to DB"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    process_dogs(args.file, drop_existing=args.drop, smoke_test=args.smoke_test, N=int(args.N))
    logging.info("Embeddings processing completed.")


if __name__ == "__main__":
    main()
