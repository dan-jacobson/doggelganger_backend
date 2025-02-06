import asyncio
import aiohttp
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Callable
from queue import Queue
import time

from doggelganger.utils import load_model
from doggelganger.embed import load_metadata

DATA_PATH = Path("data/dogs_20250106_105941.jsonl")
N = 1000
BATCH_SIZE = 32

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AsyncDogDataset(Dataset):
    def __init__(
        self,
        metadata: list,
        field_to_embed: str = "primary_photo_cropped",
        batch_size: int = 32,
        queue_size: int = 1000,
        num_fetchers: int = 8
    ):
        self.metadata = metadata
        self.field_to_embed = field_to_embed
        self.batch_size = batch_size
        self.image_queue = asyncio.Queue(maxsize=queue_size)
        self.processed_queue = Queue()
        self.num_fetchers = num_fetchers
        self.total_items = len(metadata)
        self.fetch_pbar = tqdm(total=self.total_items, desc="Fetching images", position=0)
        self.embed_pbar = tqdm(total=self.total_items, desc="Processing embeddings", position=1)
        print("\n")

    async def fetch_image(self, session: aiohttp.ClientSession, item: dict) -> tuple[dict, bytes | None]:
        url = item[self.field_to_embed]
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
                chunk = self.metadata[i:i + chunk_size]
                tasks = [self.fetch_image(session, item) for item in chunk]

                for task in asyncio.as_completed(tasks):
                    item, image_data = await task
                    if image_data:
                        try:
                            image = Image.open(BytesIO(image_data))
                            await self.image_queue.put((item, image))
                            self.fetch_pbar.update(1)
                        except Exception as e:
                            logging.error(f"Error processing image: {e}")

        # stop signal for producer: we've processed all the images
        await self.image_queue.put(None)
        self.fetch_pbar.close()

    async def consumer(self, model: torch.nn.Module):
        current_batch_images = []
        current_batch_metadata = []

        while True:
            try:
                item = await self.image_queue.get()

                if item is None:
                    if current_batch_images:
                        embeddings = model(current_batch_images, batch_size = len(current_batch_images))
                        self.processed_queue.put((embeddings, current_batch_metadata))
                        self.embed_pbar.update(len(current_batch_images))
                    break

                metadata, image = item
                current_batch_images.append(image)
                current_batch_metadata.append(metadata)

                if len(current_batch_images) >= self.batch_size:
                    embeddings = model(current_batch_images, batch_size=len(current_batch_images))
                    self.processed_queue.put((embeddings, current_batch_metadata))
                    self.embed_pbar.update(len(current_batch_images))
                    current_batch_images = []
                    current_batch_metadata = []
            
            except Exception as e:
                logging.error(f"Error in consumer: {e}")
                continue

        # stop signal for consumer
        self.processed_queue.put(None)
        self.embed_pbar.close()

    async def process_all(self, model: torch.nn.Module):
        producer_task = asyncio.create_task(self.producer())
        consumer_task = asyncio.create_task(self.consumer(model))

        await asyncio.gather(producer_task, consumer_task)

        embeddings = []
        metadata_processed = []

        while True:
            result = self.processed_queue.get()
            if result is None:
                break
            emb, meta = result
            embeddings.append(emb)
            metadata_processed.extend(meta)

        return embeddings, metadata_processed

    def __call__(self, model: torch.nn.Module):
        """Makes the dataset callable, trying to make this flow a bit more pythonic"""
        return asyncio.run(self.process_all(model))

def main():
    start_time = time.time()


    model = load_model(device="mps")
    metadata = load_metadata(DATA_PATH)
    logging.info(f"Processing metadata file: {DATA_PATH}")
    logging.info(f"Number of dogs found in file: {len(metadata)}")

    if N:
        metadata = metadata[:N]
        logging.info(f"Smoke test: Reducing number of dogs to: {len(metadata)}")

    # I was tryna make this feel pythonic, and dataset(model) was as good as I could come up with
    dataset = AsyncDogDataset(metadata=metadata, batch_size=BATCH_SIZE)
    embeddings, metadata_processed = dataset(model)

    logging.info(f"Processing complete. Embeddings: {len(embeddings)}, Metadata: {len(metadata_processed)}")

    end_time = time.time()
    duration = end_time - start_time

    logging.info("=== Benchmark Results ===")
    logging.info(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total duration: {duration:.2f} seconds")
    logging.info(f"Total dogs processed: {len(metadata_processed)}")
    logging.info(f"Successfully processed: {len(embeddings)}")
    logging.info(f"Average time per dog: {duration / len(metadata_processed):.2f} seconds")
    logging.info(f"Dogs per second: {len(metadata_processed) / duration:.2f}")

if __name__ == "__main__":
    main()