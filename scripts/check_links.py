import json
import asyncio
import aiohttp
import argparse
from tqdm.asyncio import tqdm
import random
import vecs
import os
from dotenv import load_dotenv

DOG_FILE = 'data/petfinder/dog_metadata.json'
load_dotenv()
DB_CONNECTION = os.getenv("SUPABASE_DB")

async def check_link(session, dog, is_retry=False, max_retries=3):
    url = dog["adoption_link"]
    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=10, allow_redirects=True) as response:
                success = response.status == 200
                if success and not is_retry:
                    # Check image_url if adoption_link is successful
                    image_url = dog.get("image_url")
                    if image_url:
                        async with session.get(image_url, timeout=10, allow_redirects=True) as img_response:
                            img_success = img_response.status == 200
                            if img_success:
                                # Check if the Content-Type is an image
                                content_type = img_response.headers.get("Content-Type", "")
                                img_success = content_type.startswith("image/")
                            return success, img_success, dog
                return success, None, dog
        except (asyncio.TimeoutError, aiohttp.ClientError):
            if attempt == max_retries - 1:
                return False, None, dog
            await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
    return False, None, dog

async def check_links(dogs, is_retry=False):
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
        async def check_with_semaphore(dog):
            async with semaphore:
                return await check_link(session, dog, is_retry)
        tasks = [check_with_semaphore(dog) for dog in dogs]
        return await tqdm.gather(*tasks, desc="Checking links", total=len(dogs))

def get_dogs_from_db():
    vx = vecs.create_client(DB_CONNECTION)
    dogs = vx.get_collection("dog_embeddings")
    return [record.metadata for record in dogs.peek(limit=None)]

async def main():
    parser = argparse.ArgumentParser(description="Check adoption links and image URLs.")
    parser.add_argument("-N", type=int, help="Number of links to check.", default=None)
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove dogs where either the adoption link or image failed to load.",
    )
    parser.add_argument(
        "--source",
        choices=["file", "db"],
        default="file",
        help="Source of dog data (file or database)",
    )
    args = parser.parse_args()

    if args.source == "file":
        # Load the JSON file
        with open(DOG_FILE, "r") as f:
            dogs = json.load(f)
    else:
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

    print(f"1. Number of successes: {successes}")
    print(f"2. Number of failures: {failures}")
    print(f"3. Number of initial failures that worked the second time: {retry_successes}")
    print(f"4. Number of successes whose 'image_url' didn't work: {image_failures}")
    print(f"5. Percent successes: {success_percent:.2f}%")
    print(f"6. Number of dogs removed: {removed}")

    print("\n5 examples of successes whose 'image_url' didn't work:")
    for i, dog in enumerate(image_failure_examples, 1):
        print(f"Example {i}:")
        print(f"  Name: {dog.get('name', 'N/A')}")
        print(f"  Adoption Link: {dog.get('adoption_link', 'N/A')}")
        print(f"  Image URL: {dog.get('image_url', 'N/A')}")
        print()

    if args.remove:
        if args.source == "file":
            # Save the updated JSON file with only valid dogs
            with open(DOG_FILE, "w") as f:
                json.dump(valid_dogs, f, indent=2)
            print(f"Updated {DOG_FILE} with {len(valid_dogs)} valid dogs.")
        else:
            # Update the database with only valid dogs
            vx = vecs.create_client(DB_CONNECTION)
            dogs = vx.get_collection("dog_embeddings")
            
            # Remove all documents
            dogs.delete()
            
            # Insert valid dogs
            if valid_dogs:
                records = [(dog['id'], dog['embedding'], dog) for dog in valid_dogs]
                dogs.upsert(records)
            
            print(f"Updated database with {len(valid_dogs)} valid dogs.")

if __name__ == "__main__":
    asyncio.run(main())
