import logging
import time
import asyncio
import aiohttp
from collections.abc import Callable
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import jsonlines
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

from doggelganger.utils import load_model

DATA_PATH = Path("data/dogs_20250106_105941.jsonl")
BATCH_SIZE = 64
NUM_WORKERS = 0
N = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def fetch_image(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """Fetch a single image from a URL."""
    async with semaphore:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.read()
                    return {
                        "success": True,
                        "image": Image.open(BytesIO(data))
                    }
                return {"success": False, "image": None}
        except Exception as e:
            logging.error(f"Error downloading {url}: {e}")
            return {"success": False, "image": None}

async def fetch_all_images(urls: List[str], max_concurrent: int = 50) -> List[Dict[str, Any]]:
    """Fetch multiple images concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image(session, url, semaphore) for url in urls]
        return await async_tqdm.gather(*tasks, desc="Downloading images")


class DogDataset(Dataset):
    def __init__(
        self,
        images_with_metadata: List[Dict[str, Any]],
        transform: Callable | None = None,
    ):
        """
        Initialize dataset with pre-fetched images.
        images_with_metadata: List of dicts containing 'image' and 'metadata' keys
        """
        self.data = images_with_metadata
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item


def collate_fn(batch):
    """Collate batch of images and metadata."""
    return {
        "images": [item["image"] for item in batch],
        "metadata": [item["metadata"] for item in batch]
    }

def load_metadata(metadata_path):
    """Load metadata from either JSON or JSONL file."""
    with jsonlines.open(metadata_path) as reader:
        return list(reader)

async def fetch_images_for_metadata(metadata, field_to_embed="primary_photo_cropped"):
    """Fetch all images for the given metadata."""
    urls = [item[field_to_embed] for item in metadata]
    results = await fetch_all_images(urls)
    
    # Combine images with metadata
    images_with_metadata = []
    for result, meta in zip(results, metadata):
        if result["success"]:
            images_with_metadata.append({
                "image": result["image"],
                "metadata": meta
            })
    
    return images_with_metadata

def main():
    start_time = time.time()

    # Load metadata
    metadata = load_metadata(DATA_PATH)
    logging.info(f"Processing metadata file: {DATA_PATH}")
    logging.info(f"Number of dogs found in file: {len(metadata)}")

    dogs = metadata[:N]
    logging.info(f"Smoketest -- Reducing number of dogs to: {len(dogs)}")

    # Fetch all images first
    images_with_metadata = asyncio.run(fetch_images_for_metadata(dogs))

    image_end_time = time.time()
    logging.info(f"Successfully downloaded {len(images_with_metadata)} images")
    logging.info(f"Time to download {len(images_with_metadata)} images: {image_end_time - start_time}")

    # Create dataset with pre-fetched images
    dataset = DogDataset(images_with_metadata=images_with_metadata)
    
    # Load model
    pipe = load_model(device="mps")

    # Process through model
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    embeddings = []
    for batch in tqdm(dataloader, desc="Processing images"):
        if not batch:
            continue

        emb = pipe(batch["images"], batch_size=BATCH_SIZE)
        embeddings.append(emb)
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Time to process {len(dataset)} images: {end_time - image_end_time}\n")

    logging.info("=== Benchmark Results ===")
    logging.info(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total duration: {duration:.2f} seconds")
    logging.info(f"Total dogs processed: {len(dataset)}")
    logging.info(f"Successfully processed: {len(embeddings)}")
    logging.info(f"Average time per dog: {duration/len(dataset):.2f} seconds")
    logging.info(f"Dogs per second: {len(dataset)/duration:.2f}")

    logging.info("Embeddings processing completed.")


if __name__ == "__main__":
    main()
