import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import vecs
from dotenv import load_dotenv
from litestar import Litestar, get, post
from litestar.datastructures import State, UploadFile
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

import supabase
from doggelganger.models import model_classes
from doggelganger.utils import HUGGINGFACE_MODEL, get_embedding, valid_link
from doggelganger.utils import load_model as load_embedding_pipeline


@dataclass
class Match:
    dog_id: str
    dog_embedding: list[float]
    selfie_embedding: list[float]
    embedding_model: str
    alignment_model: str


load_dotenv()
# Unfortunately `vecs` and `supabase` use different cxn strings
SUPABASE_FULL_URI = os.getenv("SUPABASE_DB")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_PW = os.getenv("SUPABASE_PW")

MODEL_CLASS = os.getenv("DOGGELGANGER_ALIGNMENT_MODEL")
MODEL_VERSION = os.getenv("DOGGELGANGER_ALIGNMENT_WEIGHTS")
MODEL_WEIGHTS = Path("weights/prod") / MODEL_VERSION

# Configure Logging -- I'm just using uvicorn's. I tried so many other things and they didn't work :(
logger = logging.getLogger("uvicorn.error")

# Initialize the image feature extraction pipeline
pipe = load_embedding_pipeline()
embedding_dim = pipe.model.config.hidden_size

# Initialize the alignment model
model_class = model_classes[MODEL_CLASS]
alignment_model = model_class.load(path=MODEL_WEIGHTS, embedding_dim=embedding_dim)


# This looks kinda ugly, but we basically just move the vx.create_client() and .get_collection() call into app startup
def connect_to_vecs(app: Litestar):
    if not getattr(app.state, "vx", None):
        app.state.vx = vecs.create_client(SUPABASE_FULL_URI)
        app.state.dogs = app.state.vx.get_or_create_collection(
            name="dog_embeddings", dimension=pipe.model.config.hidden_size
        )
    return cast(vecs.Client, app.state.vx), cast(vecs.Collection, app.state.dogs)


def connect_to_supabase(app: Litestar):
    if not getattr(app.state, "supabase", None):
        app.state.supabase = supabase.create_client(SUPABASE_URL, SUPABASE_PW)
    return cast(supabase.Client, app.state.supabase)


# Disconnect from vecs on app shutdown
def disconnect_from_vecs(app: Litestar):
    if getattr(app.state, "vx", None):
        app.state.vx.disconnect()


def disconnect_from_supabase(app: Litestar):
    if getattr(app.state, "supabase", None):
        app.state.supabase.disconnect()


@get(path="/")
async def health_check() -> str:
    return Response(content="healthy", status_code=HTTP_200_OK)


@post("/embed")
async def embed_image(
    state: State,
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

        embedding = get_embedding(img, pipe=pipe)
        aligned_embedding = alignment_model.predict(embedding)

        results = state.dogs.query(
            data=aligned_embedding,
            limit=3,  # Increase limit to have more options to check
            include_metadata=True,
            include_value=True,
            include_vector=True,
        )

        if not results:
            return Response(
                content={"error": "No matches found in database"},
                status_code=HTTP_404_NOT_FOUND,
            )

        # Find the first result with an image that works
        valid_result = None

        for i, (id, score, dog_embedding, metadata) in enumerate(results):
            url = metadata["primary_photo"]
            if valid_link(metadata["primary_photo"]):
                # for some reason we have to coerce to a float64 before it can be serialized back to a python object
                dog_embedding = np.array(dog_embedding, dtype=np.float64).tolist()

                # converts cosine distance to similarity
                similarity = 1 - score

                valid_result = {**metadata, "id": id, "dog_embedding": dog_embedding, "similarity": similarity}
                logger.debug(f"Valid link after {i + 1} tries: {url}")
                break
            else:
                logger.debug(f"Invalid adoption link: {url}")

        if not valid_result:
            return Response(
                content={"error": "No valid adoption links found (ask Dan to refresh the database)"},
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


@post("/log-match")
async def log_match(state: State, data: dict) -> Response:
    try:
        dog_id = data.get("dogId")
        dog_embedding = data.get("dogEmbedding")
        user_embedding = data.get("userEmbedding")

        if not all([dog_id, dog_embedding, user_embedding]):
            # figure out which field is missing
            missing_fields = [
                field
                for field, value in zip(
                    ["dogId", "dogEmbedding", "userEmbedding"],
                    [dog_id, dog_embedding, user_embedding],
                    strict=False,
                )
                if not value
            ]
            return Response(
                content={"error": f"Missing required field(s): {', '.join(missing_fields)}"},
                status_code=HTTP_400_BAD_REQUEST,
            )

        match = Match(
            dog_id=dog_id,
            dog_embedding=dog_embedding,
            selfie_embedding=user_embedding,
            embedding_model=HUGGINGFACE_MODEL,
            alignment_model=f"{MODEL_CLASS} | {MODEL_VERSION}",
        )

        _ = (
            state.supabase.table("matches")
            .insert(
                {
                    "dog_id": match.dog_id,
                    "dog_embedding": match.dog_embedding,
                    "selfie_embedding": match.selfie_embedding,
                    "embedding_model": match.embedding_model,
                    "alignment_model": match.alignment_model,
                }
            )
            .execute()
        )

        return Response(
            content={"messages": "Match logged successfully!"},
            status_code=HTTP_200_OK,
        )
    except Exception as e:
        logger.error(f"Error logging match {str(e)}")
        return Response(
            content={"error": f"Failed to log match: {str(e)}"},
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        )


app = Litestar(
    route_handlers=[embed_image, health_check, log_match],
    on_startup=[connect_to_vecs, connect_to_supabase],
    on_shutdown=[disconnect_from_vecs, disconnect_from_supabase],
)

# test via something like
# curl -i -X POST \
#   http://0.0.0.0:8000/embed \
#   -F "image=@/path/to/your/image.jpg"

# curl -iX POST http://0.0.0.0:8000/embed -F "image=@.mint/example.jpg"

# curl -iX POST http://0.0.0.0:8000/log-match -d '{"dogId": "1234", "userEmbedding": [1,2,3], "dogEmbedding": [4,5,6]}'