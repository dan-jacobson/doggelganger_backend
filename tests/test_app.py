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
from litestar.testing import TestClient
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from app import app


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def mock_image():
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def test_health_check(test_client):
    response = test_client.get("/")
    assert response.status_code == HTTP_200_OK
    assert response.text == "healthy"


@patch("app.load_embedding_pipeline")
def test_app_initialization(mock_load_pipeline):
    """Test that the app initializes correctly with all required components"""
    mock_pipe = MagicMock()
    mock_pipe.model.config.hidden_size = 768
    mock_load_pipeline.return_value = mock_pipe

    # Re-import to trigger initialization
    import app

    assert app.pipe is not None
    assert app.alignment_model is not None
    assert app.dogs is not None


def test_invalid_file_type(test_client):
    """Test handling of invalid file types"""
    # Test with text file
    text_data = b"This is not an image"
    response = test_client.post("/embed", files={"data": ("test.txt", text_data, "text/plain")})
    assert response.status_code == HTTP_400_BAD_REQUEST
    assert "Invalid file type" in response.json()["error"]

    # Test with corrupted image data
    corrupt_data = b"pretending to be a PNG\x89PNG but actually garbage"
    response = test_client.post("/embed", files={"data": ("fake.png", corrupt_data, "image/png")})
    assert response.status_code == HTTP_400_BAD_REQUEST
    assert "Could not process image file" in response.json()["error"]

    # Test with missing content type
    response = test_client.post("/embed", files={"data": ("test.jpg", b"some data", None)})
    assert response.status_code == HTTP_400_BAD_REQUEST
    assert "Could not process image file" in response.json()["error"]


@patch("app.get_embedding")
def test_embedding_error(mock_get_embedding, test_client, mock_image):
    """Test handling of embedding generation errors"""
    mock_get_embedding.side_effect = Exception("Embedding failed")

    response = test_client.post("/embed", files={"data": ("test.png", mock_image, "image/png")})
    assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
    assert "error" in response.json()


@patch("app.get_embedding")
@patch("app.dogs.query")
def test_empty_query_results(mock_query, mock_get_embedding, test_client, mock_image):
    """Test handling of empty database query results"""
    # Create a random embedding vector of the correct dimension (384)
    mock_get_embedding.return_value = np.random.randn(384)
    mock_query.return_value = []

    response = test_client.post("/embed", files={"data": ("test.png", mock_image, "image/png")})
    assert response.status_code == HTTP_404_NOT_FOUND
    assert response.json()["error"] == "No matches found in database"


@patch("app.get_embedding")
@patch("app.dogs.query")
@patch("app.valid_link")
def test_multiple_invalid_links(mock_valid_link, mock_query, mock_get_embedding, test_client, mock_image):
    """Test handling of multiple invalid adoption links"""
    mock_get_embedding.return_value = np.random.randn(384)
    mock_query.return_value = [
        ("id1", 0.1, {"primary_photo": "http://invalid1.com"}),
        ("id2", 0.2, {"primary_photo": "http://invalid2.com"}),
        ("id3", 0.3, {"primary_photo": "http://invalid3.com"}),
    ]
    mock_valid_link.return_value = False

    response = test_client.post("/embed", files={"data": ("test.png", mock_image, "image/png")})
    assert response.status_code == HTTP_404_NOT_FOUND
    assert mock_valid_link.call_count == 3 


@patch("app.get_embedding")
@patch("app.dogs.query")
@patch("app.valid_link")
def test_alignment_model_integration(mock_valid_link, mock_query, mock_get_embedding, test_client, mock_image):
    """Test the full pipeline including alignment model"""
    initial_embedding = np.random.randn(384)
    mock_get_embedding.return_value = initial_embedding
    mock_query.return_value = [("id1", 0.1, {"primary_photo": "http://valid.com"})]
    mock_valid_link.return_value = True

    response = test_client.post("/embed", files={"data": ("test.png", mock_image, "image/png")})
    assert response.status_code == HTTP_200_OK
    result = response.json()
    assert "embedding" in result
    assert np.array_equal(result["embedding"], initial_embedding.tolist())
    assert "similarity" in result["result"]
    assert 0 <= result["result"]["similarity"] <= 1


@patch("app.get_embedding")
@patch("app.dogs.query")
@patch("app.valid_link")
def test_embed_image_success(mock_valid_link, mock_query, mock_get_embedding, test_client):
    # Mock the embedding
    mock_get_embedding.return_value = np.random.randn(384)

    # Mock the query results
    mock_query.return_value = [
        ("id1", 0.1, {"primary_photo": "http://valid.com"}),
    ]

    # Mock valid_link to return True
    mock_valid_link.return_value = True

    # Create a test image
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    response = test_client.post("/embed", files={"data": ("test.png", img_byte_arr, "image/png")})

    print(f"Response content: {response.content}")
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}, content: {response.content}"
    response_json = response.json()
    assert "embedding" in response_json, f"'embedding' not found in response: {response_json}"
    assert "result" in response_json, f"'result' not found in response: {response_json}"


@patch("app.get_embedding")
@patch("app.dogs.query")
@patch("app.valid_link")
def test_embed_image_no_valid_links(mock_valid_link, mock_query, mock_get_embedding, test_client):
    # Mock the embedding
    mock_get_embedding.return_value = np.random.randn(384)

    # Mock the query results
    mock_query.return_value = [
        ("id1", 0.1, {"primary_photo": "http://invalid.com"}),
    ]

    # Mock valid_link to return False
    mock_valid_link.return_value = False

    # Create a test image
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    response = test_client.post("/embed", files={"data": ("test.png", img_byte_arr, "image/png")})

    assert response.status_code == 404
    assert "error" in response.json()
    assert response.json()["error"] == "No valid adoption links found"
