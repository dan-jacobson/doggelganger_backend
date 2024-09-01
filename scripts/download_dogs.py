import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

folder = "example_dog_images"
N = 2000


def download_image(url, folder):
    response = requests.get(url)
    if response.status_code == 200:
        filename = url.split("/")[-1]
        filepath = os.path.join(folder, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
    return None


def main():
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Get N random dog image URLS
    response = requests.get(f"https://dog.ceo/api/breeds/image/random/{N}")
    urls = response.json()["message"]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_image, url, folder) for url in urls]
        for future in as_completed(futures):
            filepath = future.result()
            if filepath:
                print(f"Downloaded: {filepath}")


if __name__ == "__main__":
    main()
