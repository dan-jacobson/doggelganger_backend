from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
import vecs
import os
from dotenv import load_dotenv

from train import load_model, get_embedding

load_dotenv()  # Load environment variables from .env file
DB_CONNECTION = os.getenv("SUPABASE_DB")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL")

app = FastAPI()

# Initialize the image feature extraction pipeline
model, processor, device = load_model()

# Initialize vecs client
vx = vecs.create_client(DB_CONNECTION)
dogs = vx.get_or_create_collection(
    name="dog_embeddings",
    dimension=768,  # TODO (drj): figure out how to keep these in sync
)


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
            limit=1,
            include_metadata=True,
            include_value=True, # score only comes back when include_value=True
        )

        # TODO (drj): check if the links to the adoption page still work?

        # Format results
        formatted_results = [
            {
                "id": id,
                "similarity": 1 - score, # converts cosine distance to similarity
                "metadata": metadata,
            }
            for (id, score, metadata) in results
        ]
        

        return JSONResponse(
            content={
                "message": "Image processed successfully",
                "embedding": embedding.tolist(),
                "similar_images": formatted_results,
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
