from typing import Annotated
import logging
import argparse

from litestar import Litestar, get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.response import Response
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
import io
from PIL import Image
import vecs
import os
import requests
from dotenv import load_dotenv

from doggelganger.utils import get_embedding, load_model as load_embedding_pipeline
from doggelganger.train import model_classes

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
DB_CONNECTION = os.getenv("SUPABASE_DB")

# Initialize the image feature extraction pipeline
pipe = load_embedding_pipeline()

# Initialize vecs client
vx = vecs.create_client(DB_CONNECTION)
dogs = vx.get_or_create_collection(
    name="dog_embeddings", dimension=pipe.model.config.hidden_size
)

def align_embedding(embedding, model):
    return model.predict(embedding)


def is_valid_link(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


@get(path="/")
async def health_check() -> str:
    return "healthy"


@post("/embed")
async def embed_image(
    data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
) -> Response:
    try:
        logger.debug(f"Received file: {data.filename}")
        # Read the image file
        contents = await data.read()
        logger.debug(f"File size: {len(contents)} bytes")
        img = Image.open(io.BytesIO(contents))
        logger.debug(f"Image size: {img.size}")

        # Extract features
        embedding = get_embedding(img, pipe=pipe)

        # Align embedding
        aligned_embedding = align_embedding(embedding, alignment_model)

        # Query similar images
        results = dogs.query(
            data=aligned_embedding,
            limit=10,  # Increase limit to have more options to check
            include_metadata=True,
            include_value=True,
        )

        # Find the first result with a valid adoption link
        valid_result = None
        for i, (id, score, metadata) in enumerate(results):
            if is_valid_link(metadata.get("adoption_link", "")):
                valid_result = {
                    "id": id,
                    "similarity": 1 - score,  # converts cosine distance to similarity
                    "metadata": metadata,
                }
                logger.debug(
                    f"Valid link after {i + 1} tries: {metadata.get("adoption_link")}"
                )
                break
            else:
                logger.debug(f"Invalid adoption link: {metadata.get("adoption_link")}")

        if valid_result is None:
            return Response(
                content={"error": "No valid adoption links found"},
                status_code=HTTP_404_NOT_FOUND,
            )

        return Response(
            content={
                "message": "Image processed successfully",
                "embedding": embedding,
                "similar_image": valid_result,
            },
            status_code=HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content={"error": str(e)}, status_code=HTTP_500_INTERNAL_SERVER_ERROR
        )


app = Litestar([embed_image, health_check])


def main():
    import uvicorn
    parser = argparse.ArgumentParser(
        description="Serve the Doggelganger backend as a uvicorn app."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "xgboost", "resnet"],
        default="linear",
        help="Model type to use (default: linear)",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="weights/linear.json",
        help="Path to alignment model weights (default: weights/linear.json)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to pass to uvicorn (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to pass to uvicorn (default: 8000)",
    )
    args = parser.parse_args()

    model_class = model_classes[args.model]
    global alignment_model
    alignment_model = model_class.load(args.weights_path)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

# test via something like
# curl -i -X POST \
#   http://0.0.0.0:8000/embed \
#   -F "image=@/path/to/your/image.jpg"
