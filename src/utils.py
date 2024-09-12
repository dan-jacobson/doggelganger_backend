from pathlib import Path

from PIL import Image
from transformers import pipeline
from huggingface_hub import snapshot_download

HUGGINGFACE_MODEL = "facebook/dinov2-small"

def get_model():
    snapshot_download(
    repo_id=HUGGINGFACE_MODEL,
    local_dir=".",
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
        print(f"Error processing image: {str(e)}")
        return None
