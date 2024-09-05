import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def check_link(dog):
    url = dog['adoption_link']
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200, dog
    except requests.RequestException:
        return False, dog

def main():
    # Load the JSON file
    with open('data/petfinder/dog_metadata.json', 'r') as f:
        dogs = json.load(f)

    successes = 0
    failures = 0
    failed_images = []

    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_dog = {executor.submit(check_link, dog): dog for dog in dogs}
        for future in tqdm(as_completed(future_to_dog), total=len(dogs), desc="Checking adoption links"):
            is_success, dog = future.result()
            if is_success:
                successes += 1
            else:
                failures += 1
                failed_images.append(dog['local_image'])

    total = successes + failures
    success_percent = (successes / total) * 100 if total > 0 else 0

    print(f"1. Count of successes: {successes}")
    print(f"2. Count of failures: {failures}")
    print(f"3. Percent successes: {success_percent:.2f}%")
    print("4. List of 'local_image' for failures:")
    for image in failed_images:
        print(f"   - {image}")

if __name__ == "__main__":
    main()
