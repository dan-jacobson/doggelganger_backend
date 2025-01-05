import io
from unittest.mock import patch

import pytest
from litestar.testing import TestClient
from PIL import Image

from app import app
from doggelganger.utils import valid_link


@pytest.fixture
def test_client():
    return TestClient(app)


def test_health_check(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.text == "healthy"


@patch("doggelganger.serve.get_embedding")
@patch("doggelganger.serve.dogs.query")
@patch("doggelganger.serve.is_valid_link")
def test_embed_image_success(mock_is_valid_link, mock_query, mock_get_embedding, test_client):
    # Mock the embedding
    mock_get_embedding.return_value = [0.1, 0.2, 0.3]

    # Mock the query results
    mock_query.return_value = [
        (0, "id1", 0.1, {"adoption_link": "http://valid.com"}),
    ]

    # Mock is_valid_link to return True
    mock_is_valid_link.return_value = True

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
    assert "similar_image" in response_json, f"'similar_image' not found in response: {response_json}"


@patch("doggelganger.serve.get_embedding")
@patch("doggelganger.serve.dogs.query")
@patch("doggelganger.serve.is_valid_link")
def test_embed_image_no_valid_links(mock_is_valid_link, mock_query, mock_get_embedding, test_client):
    # Mock the embedding
    mock_get_embedding.return_value = [0.1, 0.2, 0.3]

    # Mock the query results
    mock_query.return_value = [
        ("id1", 0.1, {"adoption_link": "http://invalid.com"}),
    ]

    # Mock is_valid_link to return False
    mock_is_valid_link.return_value = False

    # Create a test image
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    response = test_client.post("/embed", files={"data": ("test.png", img_byte_arr, "image/png")})

    assert response.status_code == 404
    assert "error" in response.json()
    assert response.json()["error"] == "No valid adoption links found"


def test_is_valid_link():
    assert is_valid_link("https://www.google.com")
    assert not is_valid_link("https://thisisnotarealwebsite12345.com")
