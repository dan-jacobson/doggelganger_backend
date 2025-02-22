import asyncio
import logging
from pathlib import Path
from datetime import datetime

from doggelganger.scrape import PetfinderScraper
from doggelganger.embeddings import process_dogs

def refresh_db():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output_file = Path(f"data/dogs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Starting dog scraping...")
    scraper = PetfinderScraper()
    asyncio.run(scraper.scrape_all_pets(output_path=str(output_file)))

    logging.info("Starting embedding process...")
    process_dogs(output_file, drop_existing=True, N=50_000)

    logging.fino("Dog database update complete!")

if __name__ == "__main__":
    refresh_db()