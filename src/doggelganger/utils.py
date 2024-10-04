from pathlib import Path

from PIL import Image
from transformers import pipeline
from huggingface_hub import snapshot_download

HUGGINGFACE_MODEL = "facebook/dinov2-small"
# HUGGINGFACE_MODEL = "/Users/drj/.cache/huggingface/hub/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415"


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
