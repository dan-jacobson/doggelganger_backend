#!/usr/bin/env python3

import argparse
import json
import logging
import random
import re
import urllib.request
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_safe_filename(entry):
    """Create a safe filename from entry data."""
    # Extract fields, defaulting to 'unknown' if missing
    name = entry.get("name", "unknown")
    breed = entry.get("breed", "unknown")
    location = entry.get("location", {})
    city = location.get("city", "unknown")

    # Create base filename and sanitize it
    base = f"{name}_{breed}_{city}"
    # Replace unsafe chars with underscore
    safe = re.sub(r"[^\w\-_.]", "_", base)
    return safe.lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Create a carousel of dog images")
    parser.add_argument("file", type=str, help="Path to the JSONL file")
    parser.add_argument("--N", type=int, default=20, help="Number of images to select")
    return parser.parse_args()


def read_jsonl(file_path):
    entries = []
    with open(file_path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries


def download_image(url: str, output_dir: Path, entry: dict) -> Path:
    """Download an image from URL to output_dir with filename based on entry data.

    Args:
        url: The URL of the image to download
        output_dir: Path object for the output directory
        entry: Dictionary containing dog metadata

    Returns:
        Path object for the downloaded file
    """
    # Create filename from entry data
    base_filename = create_safe_filename(entry)

    # Get extension from URL, defaulting to .jpg
    url_path = Path(url.split("?")[0])  # Remove query parameters
    ext = url_path.suffix or ".jpg"

    # Create output path
    output_path = output_dir / f"{base_filename}{ext}"

    # Download the image
    urllib.request.urlretrieve(url, output_path)

    return output_path


def main():
    args = parse_args()
    logger.info(f"Starting process with N={args.N}")

    # Delete and recreate output directory
    output_dir = Path("data/carousel")
    if output_dir.exists():
        for file in output_dir.iterdir():
            file.unlink()  # Delete all files in directory
        output_dir.rmdir()  # Delete the directory
    output_dir.mkdir(parents=True)

    # Read entries
    entries = read_jsonl(args.file)
    logger.info(f"Read {len(entries)} entries from {args.file}")
    random.shuffle(entries)  # Shuffle in place

    # Process entries until we get N successful downloads
    processed_entries = []
    entry_index = 0

    while len(processed_entries) < args.N and entry_index < len(entries):
        entry = entries[entry_index]
        entry_index += 1

        logger.info(f"Processing entry {entry_index}/{len(entries)}")
        if "primary_photo" in entry:
            url = entry["primary_photo"]
            logger.debug(f"Found photo URL: {url}")
            try:
                # Download the image with entry data
                local_path = download_image(url, output_dir, entry)
                logger.info(f"Successfully downloaded to {local_path}")

                # Update the entry with local path
                entry["primary_photo"] = local_path.name
                processed_entries.append(entry)
                logger.info(f"Progress: {len(processed_entries)}/{args.N} images")
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                continue

    # Save processed entries
    output_file = output_dir / "dog_metadata.jsonl"
    with open(output_file, "w") as f:
        for entry in processed_entries:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
