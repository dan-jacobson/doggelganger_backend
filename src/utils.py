from pathlib import Path

from PIL import Image
from transformers import pipeline

HUGGINGFACE_MODEL = "weights/dinov2-small"


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
