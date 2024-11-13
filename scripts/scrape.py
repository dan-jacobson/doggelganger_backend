import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime

import aiohttp
from aiohttp import ClientSession, TCPConnector
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


@dataclass
class Animal:
    """Data class to store relevant animal fields"""

    id: str
    name: str
    breed: str
    age: str
    sex: str
    location: dict
    description: str
    url: str
    primary_photo: str
    primary_photo_cropped: str
    photo_urls: list[str]


class PetfinderScraper:
    def __init__(self):
        self.token: str | None = None
        self.token_timestamp: float | None = None
        self.rate_limit = 50  # Per the API docs
        self.token_expiry = 3600  # Per the API docs
        self.total_pets = 0
        self.collected_pets: list[Animal] = []

    async def get_new_token(self) -> str:
        """Get a new token using Selenium and CDP"""
        logging.info("Getting new token...")
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)
        
        try:
            # Store the requests
            request_log = []
            
            # Create CDP listener for network requests
            def network_listener(driver):
                def log_request(requestId, request):
                    request_log.append(request)
                
                driver.execute_cdp_cmd('Network.enable', {})
                return log_request
                
            # Add event listener using CDP
            driver.execute_cdp_cmd('Network.enable', {})
            driver.execute_cdp_cmd('Network.setRequestInterception', {'patterns': [{'urlPattern': '*'}]})
            
            # Navigate to the page
            driver.get('https://www.petfinder.com/search/dogs-for-adoption/us/ny/brooklyn/')
            time.sleep(5)  # Wait for network activity
            
            # Get all network requests
            logs = driver.get_log('performance')
            
            # Process logs to find token
            token = None
            for entry in logs:
                try:
                    # Parse log entry
                    network_log = json.loads(entry['message'])['message']
                    
                    # Look for request with token
                    if ('params' in network_log and 
                        'request' in network_log['params'] and 
                        'url' in network_log['params']['request']):
                        
                        url = network_log['params']['request']['url']
                        if 'token=' in url and 'petfinder.com' in url:
                            token = url.split('token=')[1].split('&')[0]
                            break
                except:
                    continue
            
            if token:
                self.token = token
                self.token_timestamp = time.time()
                logging.info("New token acquired")
                return token
            else:
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
            description=animal.get("description", ""),
            location={
                "city": location.get("address", {}).get("city", ""),
                "state": location.get("address", {}).get("state", ""),
                "postcode": location.get("address", {}).get("postcode", ""),
            },
            url=animal.get("social_sharing", {}).get("email_url", ""),
        )

    async def fetch_page(self, session: ClientSession, page: int) -> list[Animal]:
        """Fetch and parse a single page of results"""
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
            "accept-language: en-US,en;q=0.9",
            "priority: u=1, i",
            "referer: https://www.petfinder.com/search/dogs-for-adoption/us/ny/11238/",
            'sec-ch-ua: "Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
            "sec-ch-ua-mobile: ?0",
            'sec-ch-ua-platform: "macOS"',
            "sec-fetch-dest: empty",
            "sec-fetch-mode: cors",
            "sec-fetch-site: same-origin",
            "user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36", # noqa: E501
            "x-requested-with: XMLHttpRequest",
        }

        try:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    # Extract animals from nested JSON structure
                    try:
                        animals = data.get("result", {}).get("animals", [])
                        return [self.parse_animal_data(animal) for animal in animals]
                    except KeyError as e:
                        logging.error(f"Unexpected JSON structure: {e}")
                        logging.debug(f"Received data: {data}")
                        return []
                else:
                    logging.error(f"Error fetching page {page}: Status {response.status}")
                    return []
        except Exception as e:
            logging.error(f"Exception fetching page {page}: {str(e)}")
            return []

    async def process_batch(self, session: ClientSession, start_page: int, batch_size: int) -> list[Animal]:
        """Process a batch of pages"""
        tasks = []
        for page in range(start_page, start_page + batch_size):
            tasks.append(self.fetch_page(session, page))
            await asyncio.sleep(1 / self.rate_limit)

        results = await asyncio.gather(*tasks)
        # Flatten the list of lists into a single list of animals
        return [animal for page_results in results for animal in page_results]

    def save_progress(self):
        """Save collected pets to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pets_data_{timestamp}.json"

        # Convert dataclass objects to dictionaries
        pets_data = [
            {
                "id": pet.id,
                "name": pet.name,
                "breed": pet.breed,
                "age": pet.age,
                "location": pet.location,
                "url": pet.url,
            }
            for pet in self.collected_pets
        ]

        with open(filename, "w") as f:
            json.dump({"total_pets": self.total_pets, "pets": pets_data}, f, indent=2)

        logging.info(f"Saved {self.total_pets} pets to {filename}")
        self.collected_pets = []  # Clear memory after saving

    async def scrape_all_pets(self, save_interval: int = 1000):
        """Main function to scrape all pets"""
        logging.info("Starting pet scraping...")

        await self.get_new_token()

        async with aiohttp.ClientSession(connector=TCPConnector(limit=50)) as session:
            first_page = await self.fetch_page(session, 1)
            if not first_page:
                raise Exception("Failed to fetch first page")

            # Get the total count from the first response
            first_response = await self.fetch_page(session, 1)
            total_results = len(first_response)  # You might need to adjust this based on the actual API response
            total_pages = (total_results + 99) // 100

            logging.info(f"Total pets: {total_results}, Total pages: {total_pages}")

            batch_size = 10
            for batch_start in range(1, total_pages + 1, batch_size):
                current_batch_size = min(batch_size, total_pages - batch_start + 1)

                # Process batch
                animals = await self.process_batch(session, batch_start, current_batch_size)
                self.collected_pets.extend(animals)
                self.total_pets = len(self.collected_pets)

                logging.info(
                    f"Processed pages {batch_start}-{batch_start + current_batch_size - 1}. "
                    f"Total pets collected: {self.total_pets}"
                )

                if self.total_pets >= save_interval:
                    self.save_progress()

        self.save_progress()
        logging.info("Scraping completed!")


async def main():
    scraper = PetfinderScraper()
    await scraper.scrape_all_pets()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(main())
