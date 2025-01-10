import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from litestar.testing import AsyncTestClient
from PIL import Image

# TODO(drj): Hacky :(, fix this if we every refactor project structure
sys.path.append(str(Path(__file__).parent.parent))
from app import app, pipe, connect_to_vecs

app.debug = True

@pytest.fixture(scope="session")
async def test_client():
    mock_vx = MagicMock()
    mock_dogs = MagicMock()

    def mock_connect_to_vecs(app):
        print("Mock connect_to_vecs called!")
        app.state.vx = mock_vx
        app.state.dogs = mock_dogs
        return mock_vx, mock_dogs

    connect_to_vecs_idx = app.on_startup.index(connect_to_vecs)
    app.on_startup[connect_to_vecs_idx] = mock_connect_to_vecs

    async with AsyncTestClient(app=app) as client:
        yield client


@pytest.fixture(scope="session")
def embedding_dim():
    """Fixture to provide the model's embedding dimension"""
    return pipe.model.config.hidden_size


@pytest.fixture(scope="session")
def mock_embedding(embedding_dim):
    """Fixture to provide a consistent mock embedding of correct dimension"""
    return list(np.random.randn(embedding_dim))


@pytest.fixture(scope="session")
def mock_image():
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


async def test_health_check(test_client):
    response = await test_client.get("/")
    assert response.status_code == HTTP_200_OK
    assert response.text == "healthy"


async def test_invalid_file_type(test_client):
    """Test handling of invalid file types"""
    # Test with text file
    text_data = b"This is not an image"
    response = await test_client.post("/embed", files={"data": ("test.txt", text_data, "text/plain")})
    assert response.status_code == HTTP_400_BAD_REQUEST
    assert "Invalid file type" in response.json()["error"]

    # Test with corrupted image data
    corrupt_data = b"pretending to be a PNG\x89PNG but actually garbage"
    response = await test_client.post("/embed", files={"data": ("fake.png", corrupt_data, "image/png")})
    assert response.status_code == HTTP_400_BAD_REQUEST
    assert "Could not process image file" in response.json()["error"]

    # Test with missing content type
    response = await test_client.post("/embed", files={"data": ("test.jpg", b"some data", None)})
    assert response.status_code == HTTP_400_BAD_REQUEST
    assert "Could not process image file" in response.json()["error"]


@patch("app.get_embedding")
async def test_embedding_error(mock_get_embedding, test_client, mock_image):
    """Test handling of embedding generation errors"""
    mock_get_embedding.side_effect = Exception("Embedding failed")

    response = await test_client.post("/embed", files={"data": ("test.png", mock_image, "image/png")})
    assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
    assert "error" in response.json()


async def test_empty_query_results(test_client, mock_image):
    """Test handling of empty database query results"""
    test_client.app.state.dogs.query.return_value = []

    response = await test_client.post("/embed", files={"data": ("test.png", mock_image, "image/png")})
    assert response.status_code == HTTP_404_NOT_FOUND
    assert response.json()["error"] == "No matches found in database"


@patch("app.valid_link")
async def test_multiple_invalid_links(mock_valid_link, test_client, mock_image):
    """Test handling of multiple invalid adoption links"""
    test_client.app.state.dogs.query.return_value = [
        ("id1", 0.1, {"primary_photo": "http://invalid1.com"}),
        ("id2", 0.2, {"primary_photo": "http://invalid2.com"}),
        ("id3", 0.3, {"primary_photo": "http://invalid3.com"}),
    ]
    mock_valid_link.return_value = False

    response = await test_client.post("/embed", files={"data": ("test.png", mock_image, "image/png")})
    assert response.status_code == HTTP_404_NOT_FOUND
    assert mock_valid_link.call_count == 3


@patch("app.get_embedding")
@patch("app.valid_link")
async def test_alignment_model_integration(mock_valid_link, mock_get_embedding, test_client, mock_image, mock_embedding):
    """Test the full pipeline including alignment model"""
    mock_get_embedding.return_value = mock_embedding
    test_client.app.state.dogs.query.return_value = [("id1", 0.1, {"primary_photo": "http://valid.com"})]
    mock_valid_link.return_value = True

    response = await test_client.post("/embed", files={"data": ("test.png", mock_image, "image/png")})
    assert (
        response.status_code == HTTP_200_OK
    ), f"Unexpected status code: {response.status_code}, content: {response.content}"
    result = response.json()
    assert "embedding" in result, f"'embedding' not found in response: {result}"
    assert "result" in result, f"'result' not found in response: {result}"
    assert np.array_equal(result["embedding"], mock_embedding)
    assert "similarity" in result["result"]
    assert 0 <= result["result"]["similarity"] <= 1
