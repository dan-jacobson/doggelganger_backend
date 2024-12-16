from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image
from transformers import pipeline

HUGGINGFACE_MODEL = "facebook/dinov2-small"


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
    pipe = pipeline(task="image-feature-extraction", model=HUGGINGFACE_MODEL, pool=True)

    return pipe


def get_embedding(img, pipe):
    try:
        if isinstance(img, Path):
            img = Image.open(img)
        return pipe(img)[0]
    except Exception as e:
        print(f"Error processing image {img}: {str(e)}")
        return None
