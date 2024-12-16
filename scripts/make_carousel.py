#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path
import urllib.request
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Create a carousel of dog images')
    parser.add_argument('--file', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--N', type=int, default=10, help='Number of images to select')
    return parser.parse_args()

def read_jsonl(file_path):
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries

def download_image(url, output_dir):
    # Create a filename from the URL
    filename = os.path.basename(url.split('?')[0])  # Remove query parameters
    output_path = output_dir / filename
    
    # Download the image
    urllib.request.urlretrieve(url, output_path)
    return output_path

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path('data/carousel')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read entries
    entries = read_jsonl(args.file)
    random.shuffle(entries)  # Shuffle in place
    
    # Process entries until we get N successful downloads
    processed_entries = []
    entry_index = 0
    
    while len(processed_entries) < args.N and entry_index < len(entries):
        entry = entries[entry_index]
        entry_index += 1
        
        if 'photo' in entry and 'url' in entry['photo']:
            try:
                # Download the image
                local_path = download_image(entry['photo']['url'], output_dir)
                
                # Update the entry with local path
                entry['photo']['local_path'] = str(local_path)
                processed_entries.append(entry)
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
