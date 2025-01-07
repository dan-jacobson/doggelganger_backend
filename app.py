import io
import logging
import os
from pathlib import Path
from typing import Annotated

import vecs
from dotenv import load_dotenv
from litestar import Litestar, get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.response import Response
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from PIL import Image

from doggelganger.models import model_classes
from doggelganger.utils import get_embedding, valid_link
from doggelganger.utils import load_model as load_embedding_pipeline

load_dotenv()
DOGGELGANGER_DB_CONNECTION = os.getenv("SUPABASE_DB")
MODEL_CLASS = os.getenv("DOGGELGANGER_ALIGNMENT_MODEL")
MODEL_WEIGHTS = Path("weights/prod") / os.getenv("DOGGELGANGER_ALIGNMENT_WEIGHTS")

# Configure Logging -- I'm just using uvicorn's. I tried so many other things and they didn't work :(
logger = logging.getLogger("uvicorn.error")

# Initialize the image feature extraction pipeline
pipe = load_embedding_pipeline()
embedding_dim = pipe.model.config.hidden_size

# Initialize the alignment model
model_class = model_classes[MODEL_CLASS]
alignment_model = model_class.load(path=MODEL_WEIGHTS, embedding_dim=embedding_dim)

# Initialize vecs client
vx = vecs.create_client(DOGGELGANGER_DB_CONNECTION)
dogs = vx.get_or_create_collection(name="dog_embeddings", dimension=pipe.model.config.hidden_size)


@get(path="/")
async def health_check() -> str:
    return Response(content="healthy", status_code=HTTP_200_OK)


@post("/embed")
async def embed_image(
    data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
) -> Response:
    try:
        logger.debug(f"Received file: {data.filename}")

        # Validate file type
        if not data.content_type or not data.content_type.startswith("image/"):
            return Response(
                content={"error": "Invalid file type. Only images are accepted."},
                status_code=HTTP_400_BAD_REQUEST,
            )

        # Read the image file
        contents = await data.read()
        logger.debug(f"File size: {len(contents)} bytes")

        try:
            img = Image.open(io.BytesIO(contents))
        except Exception:
            return Response(
                content={"error": "Could not process image file. File may be corrupted or in an unsupported format."},
                status_code=HTTP_400_BAD_REQUEST,
            )
        logger.debug(f"Image size: {img.size}")

        # Extract features
        embedding = get_embedding(img, pipe=pipe)

        # Align embedding
        aligned_embedding = alignment_model.predict(embedding)

        # Query similar images
        results = dogs.query(
            data=aligned_embedding,
            limit=3,  # Increase limit to have more options to check
            include_metadata=True,
            include_value=True,
        )

        if not results:
            return Response(
                content={"error": "No matches found in database"},
                status_code=HTTP_404_NOT_FOUND,
            )

        # Find the first result with a valid adoption link
        valid_result = None
        for i, (id, score, metadata) in enumerate(results):
            url = metadata["primary_photo"]
            if valid_link(url):
                valid_result = {
                    **metadata,
                    "id": id,
                    "similarity": 1 - score,  # converts cosine distance to similarity
                }
                logger.debug(f"Valid link after {i + 1} tries: {url}")
                break
            else:
                logger.debug(f"Invalid adoption link: {url}")

        if not valid_result:
            return Response(
                content={"error": "No valid links found"},
                status_code=HTTP_404_NOT_FOUND,
            )

        return Response(
            content={
                "message": "Image processed successfully",
                "embedding": embedding,
                "result": valid_result,
            },
            status_code=HTTP_200_OK,
        )

    except Exception as e:
        return Response(content={"error": str(e)}, status_code=HTTP_500_INTERNAL_SERVER_ERROR)


app = Litestar(
    route_handlers=[embed_image, health_check],
)

# test via something like
# curl -i -X POST \
#   http://0.0.0.0:8000/embed \
#   -F "image=@/path/to/your/image.jpg"

# curl -i -X POST \
#   http://127.0.0.1:8000/embed \
#   -F "image=@.mint/example.jpg"

# curl -i -X POST http://0.0.0.0:8000/embed -F "image=@.mint/example.jpg"
