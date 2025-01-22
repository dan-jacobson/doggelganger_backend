import logging
import time
from collections.abc import Callable
from datetime import datetime
from io import BytesIO
from pathlib import Path

import jsonlines
import requests
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from doggelganger.utils import load_model

DATA_PATH = Path("data/dogs_20250106_105941.jsonl")
BATCH_SIZE = 16
NUM_WORKERS = 8
N = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DogDataset(Dataset):
    def __init__(
        self,
        jsonl_path: list | str | Path,
        transform: Callable | None = None,
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


def collate_fn(batch):
    # filter out the images that failed to fetch
    batch = [b for b in batch if b is not None]

    if not batch:
        return None

    return {"images": [item["image"] for item in batch], "metadata": [item["metadata"] for item in batch]}


def load_metadata(metadata_path):
    """Load metadata from either JSON or JSONL file."""
    with jsonlines.open(metadata_path) as reader:
        return list(reader)


def main():
    start_time = time.time()

    pipe = load_model(device="mps")

    # Load metadata
    metadata = load_metadata(DATA_PATH)
    logging.info(f"Processing metadata file: {DATA_PATH}")
    logging.info(f"Number of dogs found in file: {len(metadata)}")

    dogs = metadata[:N]
    logging.info(f"Smoketest: Reducing number of dogs to: {len(dogs)}")

    dataset = DogDataset(jsonl_path=dogs)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    embeddings = []
    for batch in tqdm(dataloader):
        if not batch:
            continue

        emb = pipe(batch["images"], batch_size=BATCH_SIZE)
        embeddings.append(emb)

    end_time = time.time()
    duration = end_time - start_time

    logging.info("=== Benchmark Results ===")
    logging.info(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total duration: {duration:.2f} seconds")
    logging.info(f"Total dogs processed: {len(dataloader)}")
    logging.info(f"Successfully processed: {len(embeddings)}")
    logging.info(f"Average time per dog: {duration / len(dataloader):.2f} seconds")
    logging.info(f"Dogs per second: {len(dataloader) / duration:.2f}")

    logging.info("Embeddings processing completed.")


if __name__ == "__main__":
    main()


#
