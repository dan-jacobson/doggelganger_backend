from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image
from transformers import pipeline

HUGGINGFACE_MODEL = "facebook/dinov2-small"


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
