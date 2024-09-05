from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

HUGGINGFACE_MODEL = "weights/dinov2-small"


def load_model():
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    processor = AutoImageProcessor.from_pretrained(HUGGINGFACE_MODEL)
    model = AutoModel.from_pretrained(HUGGINGFACE_MODEL)
    torch.compile(model)

    model.to(device)

    return model, processor, device


def get_embedding(img, processor, model, device):
    try:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            raise ValueError("Input must be either a file path or an Image object")

        inputs = processor(images=img, return_tensors="pt")
        inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.pooler_output.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None
