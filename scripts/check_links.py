import argparse
import asyncio
import logging
import os
from datetime import datetime, timedelta
from random import random

import aiohttp
import jsonlines
import vecs
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
DB_CONNECTION = os.getenv("SUPABASE_DB")


PETFINDER_API_KEY = os.getenv("PETFINDER_API_KEY")
PETFINDER_SECRET = os.getenv("PETFINDER_SECRET")


class PetfinderAPI:
    def __init__(self):
        self.access_token = None
        self.token_expires = None

    async def get_token(self, session):
        if self.access_token and datetime.now() < self.token_expires:
            return self.access_token

        async with session.post(
            "https://api.petfinder.com/v2/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": PETFINDER_API_KEY,
                "client_secret": PETFINDER_SECRET,
            },
        ) as response:
            if response.status == 200:
                data = await response.json()
                self.access_token = data["access_token"]
                self.token_expires = datetime.now() + timedelta(seconds=data["expires_in"])
                return self.access_token
            else:
                raise Exception(f"Failed to get token: {response.status}")

    async def check_animal(self, session, animal_id):
        token = await self.get_token(session)
        headers = {"Authorization": f"Bearer {token}"}

        async with session.get(f"https://api.petfinder.com/v2/animals/{animal_id}", headers=headers) as response:
            return response.status == 200


async def check_link(session, dog, petfinder_api, is_retry=False, max_retries=3):
    url = dog["url"]
    # Extract the animal ID from the URL
    try:
        animal_id = url.split("/dog/")[1].split("/")[1]
    except IndexError:
        logging.error(f"Could not extract animal ID from URL: {url}")
        return False, None, dog
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(0.5)  # Respect rate limits
            success = await petfinder_api.check_animal(session, animal_id)
            if success and not is_retry:
                # Check image_url if adoption_link is successful
                image_url = dog.get("primary_photo")
                if image_url:
                    async with session.get(image_url, timeout=10, allow_redirects=True) as img_response:
                        img_success = img_response.status == 200
                        if img_success:
                            # Check if the Content-Type is an image
                            content_type = img_response.headers.get("Content-Type", "")
                            img_success = content_type.startswith("image/")
                        return success, img_success, dog
                return success, None, dog
        except (TimeoutError, aiohttp.ClientError):
            if attempt == max_retries - 1:
                return False, None, dog
            await asyncio.sleep(2**attempt + random.uniform(0, 1))
    return False, None, dog


async def check_links(dogs, is_retry=False):
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(5)  # Reduce concurrent requests
        petfinder_api = PetfinderAPI()

        async def check_with_semaphore(dog):
            async with semaphore:
                return await check_link(session, dog, petfinder_api, is_retry)

        tasks = [check_with_semaphore(dog) for dog in dogs]
        return await tqdm.gather(
            *tasks, desc=f"{'Retrying failed links' if is_retry else 'Checking links'}", total=len(dogs)
        )


def get_dogs_from_db():
    vx = vecs.create_client(DB_CONNECTION)
    dogs = vx.get_collection("dog_embeddings")
    return [record.metadata for record in dogs.query(data=[0] * dogs.dimension)]


async def main():
    parser = argparse.ArgumentParser(description="Check adoption links and image URLs.")
    parser.add_argument("-N", type=int, help="Number of links to check.", default=None)
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove dogs where either the adoption link or image failed to load.",
    )

    # Create mutually exclusive group for data source
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--db",
        action="store_true",
        help="Use database as data source",
    )
    source_group.add_argument(
        "--file",
        type=str,
        default="data/dogs_latest.jsonl",
        help="Path to JSONL file (default: data/dogs_latest.jsonl)",
    )
    args = parser.parse_args()

    if args.file:
        if not args.file.endswith(".jsonl"):
            raise ValueError(f"File must be .jsonl. Got: {args.file.split('.')[-1]}")
        dogs = []
        with jsonlines.open(args.file) as reader:
            for dog in reader:
                dogs.append(dog)

    if args.db:
        # Load dogs from the database
        dogs = get_dogs_from_db()

    # Limit the number of dogs if N is specified
    if args.N is not None:
        dogs = dogs[: args.N]

    successes = 0
    failures = 0
    retry_successes = 0
    image_failures = 0
    image_failure_examples = []
    valid_dogs = []

    # First pass: check all links
    results = await check_links(dogs)
    failed_dogs = []
    for is_success, image_success, dog in results:
        if is_success and image_success:
            successes += 1
            valid_dogs.append(dog)
        elif is_success and not image_success:
            image_failures += 1
            if len(image_failure_examples) < 5:
                image_failure_examples.append(dog)
            if not args.remove:
                valid_dogs.append(dog)
        else:
            failures += 1
            failed_dogs.append(dog)

    # Second pass: retry failed links
    if failed_dogs:
        retry_results = await check_links(failed_dogs, is_retry=True)
        for is_success, _, dog in retry_results:
            if is_success:
                retry_successes += 1
                failures -= 1  # Decrement failures count
                if not args.remove:
                    valid_dogs.append(dog)

    total = len(dogs)
    removed = total - len(valid_dogs)
    success_percent = (len(valid_dogs) / total) * 100 if total > 0 else 0

    logging.info(
        f"1. Number of successes: {successes}"
        f"\n2. Number of failures: {failures}"
        f"\n3. Number of initial failures that worked the second time: {retry_successes}"
        f"\n4. Number of successes whose 'primary_photo' didn't work: {image_failures}"
        f"\n5. Percent successes: {success_percent:.2f}%"
    )
    if args.remove:
        logging.info(f"\n6. Number of dogs removed: {removed}")

    logging.debug("\n5 examples of successes whose 'primary_photo' didn't work:")
    for i, dog in enumerate(image_failure_examples, 1):
        logging.debug(
            f"Example {i}:"
            f"\n  Name: {dog.get('name', 'N/A')}"
            f"\n  Adoption Link: {dog.get('url', 'N/A')}"
            f"\n  Image URL: {dog.get('primary_photo', 'N/A')}"
            ""
        )

    if args.remove:
        if not args.db:
            # Save the updated JSONL file with only valid dogs
            with jsonlines.open(args.file, mode="w") as writer:
                writer.write_all(valid_dogs)
            logging.info(f"Updated {args.file} with {len(valid_dogs)} valid dogs.")
        else:
            # Update the database with only valid dogs
            vx = vecs.create_client(DB_CONNECTION)
            dogs = vx.get_collection("dog_embeddings")

            # Remove all documents
            dogs.delete()

            # Insert valid dogs
            if valid_dogs:
                records = [(dog["id"], dog["embedding"], dog) for dog in valid_dogs]
                dogs.upsert(records)

            logging.info(f"Updated database with {len(valid_dogs)} valid dogs.")


if __name__ == "__main__":
    asyncio.run(main())
