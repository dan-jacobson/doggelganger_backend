#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path
import urllib.request
import os
import re

def create_safe_filename(entry):
    """Create a safe filename from entry data."""
    # Extract fields, defaulting to 'unknown' if missing
    name = entry.get('name', 'unknown')
    breed = entry.get('breed', 'unknown')
    location = entry.get('location', {})
    city = location.get('city', 'unknown')
    
    # Create base filename and sanitize it
    base = f"{name}_{breed}_{city}"
    # Replace unsafe chars with underscore
    safe = re.sub(r'[^\w\-_.]', '_', base)
    return safe.lower()

def parse_args():
    parser = argparse.ArgumentParser(description='Create a carousel of dog images')
    parser.add_argument('--file', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--N', type=int, default=20, help='Number of images to select')
    return parser.parse_args()

def read_jsonl(file_path):
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries

def download_image(url, output_dir, entry):
    # Create filename from entry data
    base_filename = create_safe_filename(entry)
    # Get extension from URL
    ext = os.path.splitext(url.split('?')[0])[1] or '.jpg'
    filename = f"{base_filename}{ext}"
    output_path = output_dir / filename
    
    # Download the image
    urllib.request.urlretrieve(url, output_path)
    return output_path

def main():
    args = parse_args()
    print(f"Starting process with N={args.N}")
    
    # Delete and recreate output directory
    output_dir = Path('data/carousel')
    if output_dir.exists():
        for file in output_dir.iterdir():
            file.unlink()  # Delete all files in directory
        output_dir.rmdir()  # Delete the directory
    output_dir.mkdir(parents=True)
    
    # Read entries
    entries = read_jsonl(args.file)
    print(f"Read {len(entries)} entries from {args.file}")
    random.shuffle(entries)  # Shuffle in place
    
    # Process entries until we get N successful downloads
    processed_entries = []
    entry_index = 0
    
    while len(processed_entries) < args.N and entry_index < len(entries):
        entry = entries[entry_index]
        entry_index += 1
        
        print(f"Processing entry {entry_index}/{len(entries)}")
        if 'photo' in entry and 'url' in entry['photo']:
            print(f"Found photo URL: {entry['photo']['url']}")
            try:
                # Download the image with entry data
                local_path = download_image(entry['photo']['url'], output_dir, entry)
                print(f"Successfully downloaded to {local_path}")
                
                # Update the entry with local path
                entry['photo']['local_path'] = str(local_path)
                processed_entries.append(entry)
                print(f"Progress: {len(processed_entries)}/{args.N} images")
            except Exception as e:
                print(f"Error processing {entry['photo']['url']}: {e}")
                continue
    
    # Save processed entries
    output_file = output_dir / 'dog_metadata.jsonl'
    with open(output_file, 'w') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry) + '\n')

if __name__ == '__main__':
    main()
