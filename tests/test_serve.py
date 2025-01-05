import io
from unittest.mock import patch

import pytest
from litestar.testing import TestClient
from PIL import Image

from app import app


@pytest.fixture
def test_client():
    return TestClient(app)


def test_health_check(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.text == "healthy"


@patch("app.get_embedding")
@patch("app.dogs.query")
@patch("app.valid_link")
def test_embed_image_success(mock_valid_link, mock_query, mock_get_embedding, test_client):
    # Mock the embedding
    mock_get_embedding.return_value = [0.1, 0.2, 0.3]

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
    mock_get_embedding.return_value = [0.1, 0.2, 0.3]

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
