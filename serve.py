from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
import io
from PIL import Image
import vecs
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

# Initialize the image feature extraction pipeline
feature_extractor = pipeline("image-feature-extraction", model="facebook/dinov2-small")

# Initialize vecs client
vecs_client = vecs.create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)
collection = vecs_client.get_or_create_collection(name="images", dimension=384)

@app.post("/embed")
async def embed_image(image: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))

        # Extract features
        features = feature_extractor(img)
        embedding = features[0].tolist()  # Convert to list for JSON serialization

        # Query similar images
        results = collection.query(
            query_vector=embedding,
            limit=5,
            include_metadata=True
        )

        # Format results
        formatted_results = [
            {
                "id": result.id,
                "score": result.score,
                "metadata": result.metadata
            } for result in results
        ]

        return JSONResponse(content={
            "message": "Image processed successfully",
            "embedding": embedding,
            "similar_images": formatted_results
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
