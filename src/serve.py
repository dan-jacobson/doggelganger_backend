from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
import vecs
import os
from dotenv import load_dotenv
import requests

from train import load_model, get_embedding

load_dotenv()  # Load environment variables from .env file
DB_CONNECTION = os.getenv("SUPABASE_DB")

app = FastAPI()

# Initialize the image feature extraction pipeline
model, processor, device = load_model()

# Initialize vecs client
vx = vecs.create_client(DB_CONNECTION)
dogs = vx.get_or_create_collection(
    name="dog_embeddings",
    dimension=768,  # TODO (drj): figure out how to keep these in sync
)


def is_valid_link(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


@app.post("/embed")
async def embed_image(image: UploadFile = File(...)):
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
            return JSONResponse(
                content={"error": "No valid adoption links found"}, status_code=404
            )

        return JSONResponse(
            content={
                "message": "Image processed successfully",
                "embedding": embedding.tolist(),
                "similar_image": valid_result,
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
