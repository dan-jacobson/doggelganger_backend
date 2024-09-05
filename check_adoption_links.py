import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def check_link(dog, is_retry=False):
    url = dog['adoption_link']
    try:
        response = requests.head(url, timeout=5)
        success = response.status_code == 200
        if success and not is_retry:
            # Check image_url if adoption_link is successful
            image_url = dog.get('image_url')
            if image_url:
                img_response = requests.head(image_url, timeout=5)
                return success, img_response.status_code == 200, dog
        return success, None, dog
    except requests.RequestException:
        return False, None, dog

def main():
    # Load the JSON file
    with open('data/petfinder/dog_metadata.json', 'r') as f:
        dogs = json.load(f)

    successes = 0
    failures = 0
    retry_successes = 0
    image_failures = 0

    # First pass: check all links
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_dog = {executor.submit(check_link, dog): dog for dog in dogs}
        failed_dogs = []
        for future in tqdm(as_completed(future_to_dog), total=len(dogs), desc="Checking adoption links"):
            is_success, image_success, dog = future.result()
            if is_success:
                successes += 1
                if image_success is False:
                    image_failures += 1
            else:
                failures += 1
                failed_dogs.append(dog)

    # Second pass: retry failed links
    if failed_dogs:
        with ThreadPoolExecutor(max_workers=10) as executor:
            retry_futures = {executor.submit(check_link, dog, is_retry=True): dog for dog in failed_dogs}
            for future in tqdm(as_completed(retry_futures), total=len(failed_dogs), desc="Retrying failed links"):
                is_success, _, _ = future.result()
                if is_success:
                    retry_successes += 1
                    failures -= 1  # Decrement failures count

    total = successes + failures
    success_percent = (successes / total) * 100 if total > 0 else 0

    print(f"1. Number of successes: {successes}")
    print(f"2. Number of failures: {failures}")
    print(f"3. Number of initial failures that worked the second time: {retry_successes}")
    print(f"4. Number of successes whose 'image_url' didn't work: {image_failures}")
    print(f"5. Percent successes: {success_percent:.2f}%")

if __name__ == "__main__":
    main()
