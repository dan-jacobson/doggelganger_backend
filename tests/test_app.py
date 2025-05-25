import io
import os
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
from app import app, connect_to_supabase, connect_to_vecs, pipe

app.debug = True


# Mock environment variables for testing
@pytest.fixture(scope="session", autouse=True)
def mock_env_vars():
    """Mock environment variables needed for the app"""
    env_vars = {
        "SUPABASE_DB": "postgresql://mock:mock@localhost:5432/mock",
        "SUPABASE_URL": "https://mock.supabase.co",
        "SUPABASE_PW": "mock_password",
        "DOGGELGANGER_ALIGNMENT_MODEL": "ResNetModel",
        "DOGGELGANGER_ALIGNMENT_WEIGHTS": "prodv0.2.pt",
    }

    with patch.dict(os.environ, env_vars):
        yield


@pytest.fixture(scope="session")
async def test_client():
    mock_vx = MagicMock()
    mock_dogs = MagicMock()
    mock_supabase = MagicMock()

    def mock_connect_to_vecs(app):
        print("Mock connect_to_vecs called!")
        app.state.vx = mock_vx
        app.state.dogs = mock_dogs
        return mock_vx, mock_dogs

    def mock_connect_to_supabase(app):
        print("Mock connect_to_supabase called!")
        app.state.supabase = mock_supabase
        return mock_supabase

    # Replace both startup functions
    connect_to_vecs_idx = app.on_startup.index(connect_to_vecs)
    connect_to_supabase_idx = app.on_startup.index(connect_to_supabase)

    app.on_startup[connect_to_vecs_idx] = mock_connect_to_vecs
    app.on_startup[connect_to_supabase_idx] = mock_connect_to_supabase

    async with AsyncTestClient(app=app) as client:
        # Make the mock supabase accessible in tests
        client.app.state.supabase = mock_supabase
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
        ("id1", 0.1, [0.1, 0.1, 0.1], {"primary_photo": "http://invalid1.com"}),
        ("id2", 0.2, [0.2, 0.2, 0.2], {"primary_photo": "http://invalid2.com"}),
        ("id3", 0.3, [0.3, 0.3, 0.3], {"primary_photo": "http://invalid3.com"}),
    ]
    mock_valid_link.return_value = False

    response = await test_client.post("/embed", files={"data": ("test.png", mock_image, "image/png")})
    assert response.status_code == HTTP_404_NOT_FOUND
    assert mock_valid_link.call_count == 3


@patch("app.get_embedding")
@patch("app.valid_link")
async def test_alignment_model_integration(
    mock_valid_link, mock_get_embedding, test_client, mock_image, mock_embedding
):
    """Test the full pipeline including alignment model"""
    mock_get_embedding.return_value = mock_embedding
    test_client.app.state.dogs.query.return_value = [
        ("id1", 0.1, [0.1, 0.1, 0.1], {"primary_photo": "http://valid.com"})
    ]
    mock_valid_link.return_value = True

    response = await test_client.post("/embed", files={"data": ("test.png", mock_image, "image/png")})
    assert response.status_code == HTTP_200_OK, (
        f"Unexpected status code: {response.status_code}, content: {response.content}"
    )
    result = response.json()
    assert "embedding" in result, f"'embedding' not found in response: {result}"
    assert "result" in result, f"'result' not found in response: {result}"
    assert np.array_equal(result["embedding"], mock_embedding)
    assert "similarity" in result["result"]
    assert 0 <= result["result"]["similarity"] <= 1


async def test_log_match_success(test_client):
    """Test successful match logging"""
    # Setup mock response
    mock_table = MagicMock()
    mock_insert = MagicMock()

    test_client.app.state.supabase.table.return_value = mock_table
    mock_table.insert.return_value = mock_insert
    mock_insert.execute.return_value = {"data": [{"id": 1}]}

    payload = {"dogId": "test-dog-123", "dogEmbedding": [0.1, 0.2, 0.3], "userEmbedding": [0.4, 0.5, 0.6]}

    response = await test_client.post("/log-match", json=payload)

    assert response.status_code == HTTP_200_OK
    assert "Match logged successfully" in response.json()["messages"]

    # Verify the database was called correctly
    test_client.app.state.supabase.table.assert_called_once_with("matches")
    mock_table.insert.assert_called_once()

    # Check the inserted data structure
    inserted_data = mock_table.insert.call_args[0][0]
    assert inserted_data["dog_id"] == "test-dog-123"
    assert inserted_data["dog_embedding"] == [0.1, 0.2, 0.3]
    assert inserted_data["selfie_embedding"] == [0.4, 0.5, 0.6]
    assert "facebook/dinov2-small" in inserted_data["embedding_model"]
    assert inserted_data["alignment_model"]  # Should contain model class and version


async def test_log_match_missing_fields(test_client):
    """Test log match with missing required fields"""
    # Test missing dogId
    payload = {"dogEmbedding": [0.1, 0.2, 0.3], "userEmbedding": [0.4, 0.5, 0.6]}
    response = await test_client.post("/log-match", json=payload)
    assert response.status_code == HTTP_400_BAD_REQUEST
    assert "Missing required field(s): dogId" in response.json()["error"]

    # Test missing multiple fields
    payload = {"dogId": "test-123"}
    response = await test_client.post("/log-match", json=payload)
    assert response.status_code == HTTP_400_BAD_REQUEST
    error_msg = response.json()["error"]
    assert "Missing required field(s)" in error_msg
    assert "dogEmbedding" in error_msg
    assert "userEmbedding" in error_msg


async def test_log_match_database_error(test_client):
    """Test log match when database operation fails"""
    # Setup mock to raise an exception
    test_client.app.state.supabase.table.side_effect = Exception("Database connection failed")

    payload = {"dogId": "test-dog-123", "dogEmbedding": [0.1, 0.2, 0.3], "userEmbedding": [0.4, 0.5, 0.6]}

    response = await test_client.post("/log-match", json=payload)

    assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to log match" in response.json()["error"]


async def test_log_match_empty_payload(test_client):
    """Test log match with empty payload"""
    response = await test_client.post("/log-match", json={})
    assert response.status_code == HTTP_400_BAD_REQUEST
    error_msg = response.json()["error"]
    assert "Missing required field(s)" in error_msg
    assert "dogId" in error_msg
    assert "dogEmbedding" in error_msg
    assert "userEmbedding" in error_msg


@patch("app.get_embedding")
@patch("app.valid_link")
async def test_full_pipeline_with_match_logging(
    mock_valid_link, mock_get_embedding, test_client, mock_image, mock_embedding
):
    """Test the full pipeline and then log the match"""
    mock_get_embedding.return_value = mock_embedding
    test_client.app.state.dogs.query.return_value = [
        ("test-dog-123", 0.1, [0.1, 0.1, 0.1], {"primary_photo": "http://valid.com"})
    ]
    mock_valid_link.return_value = True

    # First, get a match
    response = await test_client.post("/embed", files={"data": ("test.png", mock_image, "image/png")})
    assert response.status_code == HTTP_200_OK
    result = response.json()

    # Then log the match
    match_payload = {
        "dogId": result["result"]["id"],
        "dogEmbedding": result["result"]["dog_embedding"],
        "userEmbedding": result["embedding"],
    }

    log_response = await test_client.post("/log-match", json=match_payload)
    assert log_response.status_code == HTTP_200_OK
