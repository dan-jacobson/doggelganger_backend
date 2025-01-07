import os
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import pipeline

HUGGINGFACE_MODEL = os.getenv("DOGGELGANGER_HUGGINGFACE_MODEL", "facebook/dinov2-small")


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


def download_model_weights():
    snapshot_download(
        repo_id=HUGGINGFACE_MODEL,
        allow_patterns=["*.safetensors", "*.json", "README.md"],
    )


def load_model():
    """Convenience function to load the image embedding model."""
    pipe = pipeline(task="image-feature-extraction", model=HUGGINGFACE_MODEL, pool=True)

    return pipe


def get_embedding(img, pipe):
    """Convenience function allowing pipelines to handle pathlib `Path` objects."""
    try:
        if isinstance(img, Path):
            img = Image.open(img)
        return pipe(img)[0]
    except Exception as e:
        print(f"Error processing image {img}: {str(e)}")
        return None


def valid_link(url):
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException:
        return False
