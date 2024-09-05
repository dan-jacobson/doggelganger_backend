import json
import os
import time
import argparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import requests


def sanitize_name(name):
    # Remove unwanted phrases
    unwanted_phrases = [", adoptable Dog", ", Out-of-town pet", " *foster needed*", "/"]
    for phrase in unwanted_phrases:
        name = name.replace(phrase, "").strip()

    if name.endswith("."):
        name = name.rstrip(".")
    return name


def sanitize_image_url(url):
    if "?" in url:
        url = url.split("?")[0]
    return url


def parse_name(full_name):
    parts = full_name.split(',')
    name = parts[0].strip()
    details = parts[1].strip().split()
    age = details[0]
    sex = details[1]
    breed = " ".join(details[2:])
    distance = parts[2].strip().replace(".", "") if len(parts) > 2 else ""
    
    return {
        "name": name,
        "full_name": full_name,
        "age": age,
        "sex": sex,
        "breed": breed,
        "distance": distance
    }


def get_dog_data(driver, url, city):
    try:
        print(f"Fetching URL: {url}")
        driver.get(url)

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.CLASS_NAME, "petCard.petCard_searchResult")
            )
        )

        # Scroll the page in increments
        for scroll_percentage in [0.2, 0.4, 0.6, 0.8]:
            driver.execute_script(
                f"window.scrollTo(0, document.body.scrollHeight * {scroll_percentage});"
            )
            time.sleep(2)  # Wait for content to load after each scroll

        dogs = []
        dog_elements = driver.find_elements(
            By.CLASS_NAME, "petCard.petCard_searchResult"
        )
        print(f"Found {len(dog_elements)} dog elements on the page")

        for dog_element in dog_elements:
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CLASS_NAME, "petCard-body-details-hdg")
                    )
                )
                name_element = dog_element.find_element(
                    By.CLASS_NAME, "petCard-body-details-hdg"
                )
                full_name = name_element.find_elements(By.TAG_NAME, "span")[
                    1
                ].text.strip()
                sanitized_full_name = sanitize_name(full_name)
                parsed_name = parse_name(sanitized_full_name)
                
                image_url = dog_element.find_element(By.TAG_NAME, "img").get_attribute(
                    "src"
                )
                sanitized_image_url = sanitize_image_url(image_url)

                adoption_link = dog_element.find_element(
                    By.CLASS_NAME, "petCard-link"
                ).get_attribute("href")

                if parsed_name["name"] and sanitized_image_url and adoption_link:
                    dog_data = {
                        **parsed_name,
                        "location": city,
                        "image_url": sanitized_image_url,
                        "adoption_link": adoption_link,
                    }
                    dogs.append(dog_data)
                    print(f"Processed dog: {parsed_name['name']} in {city}")
                else:
                    print("Skipped a dog due to missing information")
            except Exception as e:
                print(f"Error processing a dog element: {e}")

        return dogs
    except Exception as e:
        print(f"Error fetching or parsing the page: {e}")
        return []


def download_image(url, folder, filename):
    try:
        print(f"Downloading image from {url}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(os.path.join(folder, filename), "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            print(f"Image saved as {filename}")
            return True
        else:
            print(f"Image not found at {url}. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return False


def save_metadata(dogs, output_folder):
    metadata_file = os.path.join(output_folder, "dog_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(dogs, f, indent=2)
    print(f"Updated metadata saved to {metadata_file}")


def load_metadata(output_folder):
    metadata_file = os.path.join(output_folder, "dog_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser(description="Scrape dog data from Petfinder")
    parser.add_argument("--download", action="store_true", help="Download dog images")
    parser.add_argument("--N", type=int, default=1000, help="Number of dogs to scrape")
    args = parser.parse_args()

    cities = {
        "Brooklyn": "https://www.petfinder.com/search/dogs-for-adoption/us/ny/brooklyn/",
        "Manhattan": "https://www.petfinder.com/search/dogs-for-adoption/us/ny/manhattan/",
        "Boston": "https://www.petfinder.com/search/dogs-for-adoption/us/ma/boston/",
        "Seattle": "https://www.petfinder.com/search/dogs-for-adoption/us/wa/seattle/",
        "San Francisco": "https://www.petfinder.com/search/dogs-for-adoption/us/ca/san-francisco/",
    }

    output_folder = "data/petfinder"
    os.makedirs(output_folder, exist_ok=True)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    all_dogs = load_metadata(output_folder)
    existing_dogs = {dog["adoption_link"] for dog in all_dogs}

    try:
        for city, base_url in cities.items():
            print(f"Scraping dogs from {city}")
            page = 1
            empty_pages = 0
            max_empty_pages = 3
            city_dogs = 0

            while empty_pages < max_empty_pages and city_dogs < args.N:
                print(f"Scraping page {page}...")
                url = f"{base_url}?page={page}"

                dogs = get_dog_data(driver, url, city)
                if not dogs:
                    empty_pages += 1
                    print(f"No dogs found on this page. Empty pages: {empty_pages}")
                else:
                    empty_pages = 0  # Reset empty pages counter
                    for dog in dogs:
                        if city_dogs >= args.N:
                            break
                        if dog["adoption_link"] not in existing_dogs:
                            if args.download:
                                filename = f"{dog['name']}_{city.replace(' ', '_')}.jpg"
                                if download_image(
                                    dog["image_url"], output_folder, filename
                                ):
                                    dog["local_image"] = filename
                            all_dogs.append(dog)
                            existing_dogs.add(dog["adoption_link"])
                            city_dogs += 1
                        else:
                            print(
                                f"Skipping dog with adoption link {dog['adoption_link']} (already in metadata)"
                            )

                    # Save metadata incrementally after each page
                    save_metadata(all_dogs, output_folder)

                page += 1
                time.sleep(2)  # Basic rate limiting

            print(f"Scraped {city_dogs} dogs from {city}")

        print(f"Scraped a total of {len(all_dogs)} dogs.")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
