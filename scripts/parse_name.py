import json
import os
import shutil
import argparse

def parse_name(d: dict):
    name = d['name']
    d['full_name'] = name.replace(".", "")
    d['name'] = name.split(',')[0]
    d['age'] = name.split(",")[1].split()[0]
    d['sex'] = name.split(",")[1].split()[1]
    d['breed'] = " ".join(name.split(",")[1].split()[2:])
    d['distance'] = name.split(",")[2].replace(".", "")

    d['local_image'] = d['local_image'].replace(".", "", 1).replace(",", "").replace(" ", "_")
    return d

def update_image_paths(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                old_path = os.path.join(root, file)
                new_filename = file.replace(",", "").replace(" ", "_")
                if new_filename.count(".") >= 2:
                    new_filename = new_filename.replace(".", "", 1)
                new_path = os.path.join(root, new_filename)
                shutil.move(old_path, new_path)
                print(f"Renamed: {file} -> {new_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse dog metadata and update image paths.")
    parser.add_argument("directory", help="Directory containing images and dog_metadata.json")
    args = parser.parse_args()

    metadata_path = os.path.join(args.directory, 'dog_metadata.json')
    new_metadata_path = os.path.join(args.directory, 'new_dog_metadata.json')

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)

        new_metadata = [parse_name(dog) for dog in metadata]
        update_image_paths(args.directory)

        with open(new_metadata_path, 'w') as f:
            json.dump(new_metadata, f, indent=2)

        print(f"Updated metadata saved to {new_metadata_path}")
    except Exception as e:
        print(f"Error: {e}")
