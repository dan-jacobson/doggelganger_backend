import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

import aiohttp
import jsonlines
from aiohttp import ClientSession, TCPConnector
from selenium import webdriver
from tqdm.asyncio import tqdm

from doggelganger.utils import Animal


@dataclass
class PaginationInfo:
    """Data class to store pagination information"""

    count_per_page: int
    total_count: int
    current_page: int
    total_pages: int


class PetfinderScraper:
    def __init__(self):
        self.token: str | None = None
        self.token_timestamp: float | None = None
        self.rate_limit = 50  # Per the API docs
        self.token_expiry = 3600  # Per the API docs
        self.total_pets = 0
        self.collected_pets: list[Animal] = []
        self.seen_animals = set()  # Track unique animals

    async def get_new_token(self) -> str:
        """Get a new token using Selenium and CDP"""
        logging.debug("Getting new token...")

        options = webdriver.ChromeOptions()
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")

        # Set logging preferences properly
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
        options.add_experimental_option(
            "perfLoggingPrefs",
            {
                "enableNetwork": True,
                "enablePage": False,
            },
        )

        logging.debug("Initializing Chrome driver...")
        try:
            driver = webdriver.Chrome(options=options)
            logging.debug("Chrome driver initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Chrome driver: {str(e)}")
            raise

        try:
            # Enable CDP Network domain
            driver.execute_cdp_cmd("Network.enable", {})

            # Navigate to the page
            logging.debug("Attempting to navigate to Petfinder...")
            try:
                driver.get("https://www.petfinder.com/search/dogs-for-adoption/us/ny/brooklyn/")
                logging.debug("Navigation successful")
            except Exception as e:
                logging.error(f"Failed to navigate to page: {str(e)}")
                raise

            # Wait for network activity and log page info
            time.sleep(5)
            try:
                logging.debug(f"Current URL: {driver.current_url}")
                logging.debug(f"Page title: {driver.title}")
                logging.debug(f"Page source length: {len(driver.page_source)}")
            except Exception as e:
                logging.error(f"Failed to get page info: {str(e)}")

            # Get all network requests
            logs = driver.get_log("performance")
            logging.debug(f"Captured {len(logs)} network log entries")

            # Process logs to find token
            token = None
            for entry in logs:
                try:
                    network_log = json.loads(entry["message"])["message"]

                    if network_log["method"] == "Network.requestWillBeSent" and "request" in network_log["params"]:
                        url = network_log["params"]["request"].get("url", "")
                        if "https://www.petfinder.com/search/" in url and "token=" in url:
                            token = url.split("token=")[1].split("&")[0]
                            logging.debug("Found Petfinder token")
                            break
                except Exception as e:
                    logging.error(f"Error processing log entry: {str(e)}")

            if token:
                self.token = token
                self.token_timestamp = time.time()
                logging.debug("New token acquired")
                return token
            else:
                logging.error("No token found in network requests")
                logging.debug("Network logs captured:", logs)
                raise Exception("Token not found in network requests")

        finally:
            driver.quit()

    async def check_token(self):
        """Check if token needs refresh"""
        if (
            not self.token or not self.token_timestamp or time.time() - self.token_timestamp > self.token_expiry - 60
        ):  # Refresh 1 minute early
            await self.get_new_token()

    def parse_animal_data(self, result: dict) -> Animal:
        """Extract relevant fields from animal data"""
        animal = result.get("animal")
        location = result.get("location")
        return Animal(
            id=animal.get("id", ""),
            name=animal.get("name", ""),
            breed=animal.get("breeds_label", "Unknown"),
            age=animal.get("age", ""),
            sex=animal.get("sex", ""),
            location={
                "city": location.get("address", {}).get("city", ""),
                "state": location.get("address", {}).get("state", ""),
                "postcode": location.get("address", {}).get("postcode", ""),
            },
            description=animal.get("description", ""),
            url=animal.get("social_sharing", {}).get("email_url", ""),
            primary_photo=animal.get("primary_photo_url", ""),
            primary_photo_cropped=animal.get("primary_photo_url_cropped", ""),
            photo_urls=animal.get("photo_urls", ""),
        )

    async def fetch_page(self, session: ClientSession, page: int) -> tuple[PaginationInfo, list[Animal]]:
        """Fetch and parse a single page of results

        Returns:
            Tuple containing pagination info and list of animals
        """
        await self.check_token()

        url = "https://www.petfinder.com/search/"
        params = {
            "page": page,
            "limit[]": 100,
            "status": "adoptable",
            "token": self.token,
            "distance[]": 100,
            "type[]": "dogs",
            "include_transportable": "true",
        }

        headers = {
            "accept-language": "en-US,en;q=0.9",
            "priority": "u=1, i",
            "referer": "https://www.petfinder.com/search/dogs-for-adoption/us/ny/11238/",
            "sec-ch-ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",  # noqa: E501
            "x-requested-with": "XMLHttpRequest",
        }

        try:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    # Extract pagination and animals from nested JSON structure
                    try:
                        result = data.get("result", {})
                        pagination = result.get("pagination", {})
                        pagination_info = PaginationInfo(
                            count_per_page=pagination.get("count_per_page", 0),
                            total_count=pagination.get("total_count", 0),
                            current_page=pagination.get("current_page", 0),
                            total_pages=pagination.get("total_pages", 0),
                        )

                        animals = result.get("animals", [])
                        return pagination_info, [self.parse_animal_data(animal) for animal in animals]
                    except KeyError as e:
                        logging.error(f"Unexpected JSON structure: {e}")
                        logging.debug(f"Received data: {data}")
                        return PaginationInfo(0, 0, 0, 0), []
                else:
                    logging.error(f"Error fetching page {page}: Status {response.status}")
                    return PaginationInfo(0, 0, 0, 0), []
        except Exception as e:
            logging.error(f"Exception fetching page {page}: {str(e)}")
            return PaginationInfo(0, 0, 0, 0), []

    def sanitize_animals(self, animals: list[Animal]) -> list[Animal]:
        """Filter and sanitize animal photo fields"""
        sanitized_animals = []
        for animal in animals:
            if not animal.primary_photo_cropped and animal.primary_photo:
                animal.primary_photo_cropped = animal.primary_photo

            # Only keep animals that have at least one photo and a url
            if animal.primary_photo and animal.primary_photo_cropped and animal.url:
                sanitized_animals.append(animal)

        return sanitized_animals

    async def process_batch(self, session: ClientSession, start_page: int, batch_size: int) -> list[Animal]:
        """Process a batch of pages"""
        tasks = []
        for page in range(start_page, start_page + batch_size):
            tasks.append(self.fetch_page(session, page))
            await asyncio.sleep(1 / self.rate_limit)

        batch_results = await asyncio.gather(*tasks)
        # Unzip the pagination info and animals
        pagination_infos, animals_lists = zip(*batch_results, strict=False)

        # Flatten the list of animals
        all_animals = [animal for animals in animals_lists for animal in animals]

        # Filter and sanitize photo fields
        sanitized_animals = self.sanitize_animals(all_animals)

        return sanitized_animals

    def get_animal_signature(self, animal: Animal) -> str:
        """Create a unique signature for an animal based on name, breed, and description"""
        return f"{animal.name}|{animal.breed}|{animal.description}"

    def save_progress(self, output_path: str):
        """Save collected pets to file, filtering duplicates"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Filter and save unique pets
        unique_pets = []
        for pet in self.collected_pets:
            signature = self.get_animal_signature(pet)
            if signature not in self.seen_animals:
                self.seen_animals.add(signature)
                unique_pets.append(pet)

        # Convert pets to dictionaries and append to file
        with jsonlines.open(output_path, mode="a") as writer:
            for pet in unique_pets:
                writer.write(pet.__dict__)

        # Log progress
        pets_saved = len(unique_pets)
        self.total_pets += pets_saved
        duplicates = len(self.collected_pets) - pets_saved
        logging.debug(
            f"Saved {pets_saved} pets to {output_path} (Total: {self.total_pets}, Duplicates filtered: {duplicates})"
        )
        self.collected_pets = []  # Clear memory after saving

    async def scrape_all_pets(self, output_path: str, save_interval: int = 1000, smoke_test: bool = False):
        """Main function to scrape all pets"""
        logging.info("Starting pet scraping...")

        await self.get_new_token()

        async with aiohttp.ClientSession(connector=TCPConnector(limit=50)) as session:
            # Get pagination info from first page
            pagination_info, first_page_animals = await self.fetch_page(session, 1)
            if not pagination_info.total_count:
                raise Exception("Failed to fetch first page")

            logging.info(f"Total pets: {pagination_info.total_count}, Total pages: {pagination_info.total_pages}")

            batch_size = 10
            # If smoke test, only process first page
            total_pages = 1 if smoke_test else pagination_info.total_pages
            total_batches = (total_pages + batch_size - 1) // batch_size

            async for batch_start in tqdm(
                range(1, total_pages + 1, batch_size), total=total_batches, desc="Scraping pages"
            ):
                current_batch_size = min(batch_size, pagination_info.total_pages - batch_start + 1)

                # Process batch
                animals = await self.process_batch(session, batch_start, current_batch_size)
                self.collected_pets.extend(animals)

                logging.debug(
                    f"Processed pages {batch_start}-{batch_start + current_batch_size - 1}. "
                    f"Total pets collected: {self.total_pets}"
                )

                if len(self.collected_pets) >= save_interval:
                    self.save_progress(output_path)

        if self.collected_pets:
            self.save_progress(output_path)
        logging.info(f"Scraping completed! Total pets saved: {self.total_pets}")


async def main():
    parser = argparse.ArgumentParser(description="Scrape Petfinder for dogs")
    parser.add_argument("--smoke-test", action="store_true", help="Only scrape first page")
    parser.add_argument(
        "--output-file", default=f'data/dogs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl', help="Output file path"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()), format="%(asctime)s - %(levelname)s - %(message)s"
    )

    scraper = PetfinderScraper()
    await scraper.scrape_all_pets(output_path=args.output_file, smoke_test=args.smoke_test)


if __name__ == "__main__":
    asyncio.run(main())
