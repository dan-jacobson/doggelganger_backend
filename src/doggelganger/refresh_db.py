import asyncio
import logging
from datetime import datetime
from pathlib import Path

from doggelganger.embeddings import process_dogs
from doggelganger.scrape import PetfinderScraper


def refresh_db():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output_file = Path(f"data/dogs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Starting dog scraping...")
    scraper = PetfinderScraper()
    asyncio.run(scraper.scrape_all_pets(output_path=str(output_file)))

    # Empirically, each 10_000 dog embeddings consumes 50 Mb on-disk on postgres. Supabase caps us at 0.5 Gb, so we target about 80% util
    logging.info("Starting embedding process...")
    process_dogs(output_file, drop_existing=True, N=80_000)

    logging.info("Dog database update complete!")


if __name__ == "__main__":
    refresh_db()
