from litestar import Litestar, post
from litestar.datastructures import UploadFile
from litestar.response import Response
from litestar.status_codes import HTTP_200_OK, HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR
import io
from PIL import Image
import vecs
import os
import requests
from dotenv import load_dotenv

from utils import load_model, get_embedding

load_dotenv()
DB_CONNECTION = os.getenv("SUPABASE_DB")

# Initialize the image feature extraction pipeline
model, processor, device = load_model()

# Initialize vecs client
vx = vecs.create_client(DB_CONNECTION)
dogs = vx.get_or_create_collection(
    name="dog_embeddings",
    dimension=model.config.hidden_size
)

def is_valid_link(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

from litestar.params import UploadedFile

@post("/embed")
async def embed_image(image: UploadedFile) -> Response:
    try:
        # Read the image file
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))

        # Extract features
        embedding = get_embedding(img, model=model, processor=processor, device=device)

        # Query similar images
        results = dogs.query(
            data=embedding,
            limit=10,  # Increase limit to have more options to check
            include_metadata=True,
            include_value=True,
        )

        # Find the first result with a valid adoption link
        valid_result = None
        for id, score, metadata in results:
            if is_valid_link(metadata.get("adoption_link", "")):
                valid_result = {
                    "id": id,
                    "similarity": 1 - score,  # converts cosine distance to similarity
                    "metadata": metadata,
                }
                break

        if valid_result is None:
            return Response(
                content={"error": "No valid adoption links found"},
                status_code=HTTP_404_NOT_FOUND
            )

        return Response(
            content={
                "message": "Image processed successfully",
                "embedding": embedding.tolist(),
                "similar_image": valid_result,
            },
            status_code=HTTP_200_OK
        )

    except Exception as e:
        return Response(content={"error": str(e)}, status_code=HTTP_500_INTERNAL_SERVER_ERROR)

app = Litestar([embed_image])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
