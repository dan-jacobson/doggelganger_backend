import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import ClientSession

from doggelganger.scrape import Animal, PaginationInfo, PetfinderScraper


@pytest.fixture
def mock_animal_data():
    """Fixture providing sample animal data"""
    return {
        "animal": {
            "id": 12345,
            "name": "Buddy",
            "breeds_label": "Labrador Retriever",
            "age": "Young",
            "sex": "Male",
            "description": "Friendly dog looking for a home",
            "primary_photo_url": "http://example.com/photo.jpg",
            "primary_photo_url_cropped": "http://example.com/photo_cropped.jpg",
            "photo_urls": ["http://example.com/photo1.jpg", "http://example.com/photo2.jpg"],
            "social_sharing": {"email_url": "http://example.com/adopt/buddy"},
        },
        "location": {"address": {"city": "Brooklyn", "state": "NY", "postcode": "11238"}},
    }


@pytest.fixture
def mock_pagination_data():
    """Fixture providing sample pagination data"""
    return {"count_per_page": 100, "total_count": 1000, "current_page": 1, "total_pages": 10}


@pytest.fixture
def mock_api_response(mock_animal_data, mock_pagination_data):
    """Fixture providing a mock API response"""
    return {"result": {"pagination": mock_pagination_data, "animals": [mock_animal_data]}}


class MockResponse:
    """Mock aiohttp response"""

    def __init__(self, data, status=200):
        self.data = data
        self.status = status

    async def json(self):
        return self.data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.mark.asyncio
async def test_parse_animal_data(mock_animal_data):
    """Test parsing animal data from API response"""
    scraper = PetfinderScraper()
    animal = scraper.parse_animal_data(mock_animal_data)

    assert isinstance(animal, Animal)
    assert animal.id == '12345'
    assert animal.name == "Buddy"
    assert animal.breed == "Labrador Retriever"
    assert animal.age == "Young"
    assert animal.sex == "Male"
    assert animal.description == "Friendly dog looking for a home"
    assert animal.url == "http://example.com/adopt/buddy"
    assert animal.primary_photo == "http://example.com/photo.jpg"
    assert animal.primary_photo_cropped == "http://example.com/photo_cropped.jpg"
    assert animal.location["city"] == "Brooklyn"
    assert animal.location["state"] == "NY"


@pytest.mark.asyncio
@patch("doggelganger.scrape.PetfinderScraper.check_token")
async def test_fetch_page(mock_check_token):
    """Test fetching a page of results"""
    mock_check_token.return_value = None

    scraper = PetfinderScraper()
    scraper.token = "test_token"
    scraper.token_timestamp = 1000000000  # Set a timestamp that won't expire

    mock_response = {
        "result": {
            "pagination": {"count_per_page": 100, "total_count": 1000, "current_page": 1, "total_pages": 10},
            "animals": [
                {
                    "animal": {
                        "id": 12345,
                        "name": "Buddy",
                        "breeds_label": "Labrador Retriever",
                        "age": "Young",
                        "sex": "Male",
                        "description": "Friendly dog",
                        "primary_photo_url": "http://example.com/photo.jpg",
                        "primary_photo_url_cropped": "http://example.com/photo_cropped.jpg",
                        "photo_urls": ["http://example.com/photo1.jpg"],
                        "social_sharing": {"email_url": "http://example.com/adopt/buddy"},
                    },
                    "location": {"address": {"city": "Brooklyn", "state": "NY", "postcode": "11238"}},
                }
            ],
        }
    }

    # Create a mock session
    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get.return_value = MockResponse(mock_response)

    # Call the method
    pagination_info, animals = await scraper.fetch_page(mock_session, 1)

    # Verify the results
    assert isinstance(pagination_info, PaginationInfo)
    assert pagination_info.total_count == 1000
    assert pagination_info.total_pages == 10
    assert len(animals) == 1
    assert animals[0].name == "Buddy"

    # Verify the session was called with correct parameters
    mock_session.get.assert_called_once()
    args, kwargs = mock_session.get.call_args
    assert args[0] == "https://www.petfinder.com/search/"
    assert kwargs["params"]["page"] == 1
    assert kwargs["params"]["token"] == "test_token"


@pytest.mark.asyncio
@patch("doggelganger.scrape.PetfinderScraper.get_new_token")
async def test_fetch_page_error(mock_get_token):
    """Test handling errors when fetching a page"""
    mock_get_token.return_value = "test_token"

    scraper = PetfinderScraper()
    scraper.token_timestamp = 1000000000

    # Create a mock session that returns an error
    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get.return_value = MockResponse({}, status=500)

    # Call the method
    pagination_info, animals = await scraper.fetch_page(mock_session, 1)

    # Verify the results
    assert pagination_info.total_count == 0
    assert len(animals) == 0


@pytest.mark.asyncio
async def test_sanitize_animals():
    """Test sanitizing animal data"""
    scraper = PetfinderScraper()

    # Create test animals
    animals = [
        # Valid animal with all fields
        Animal(
            id=1,
            name="Buddy",
            breed="Lab",
            age="Young",
            sex="Male",
            location={},
            description="",
            url="http://example.com",
            primary_photo="http://example.com/photo.jpg",
            primary_photo_cropped="http://example.com/photo_cropped.jpg",
            photo_urls=[],
        ),
        # Animal missing cropped photo but has primary photo
        Animal(
            id=2,
            name="Max",
            breed="Poodle",
            age="Adult",
            sex="Male",
            location={},
            description="",
            url="http://example.com",
            primary_photo="http://example.com/photo.jpg",
            primary_photo_cropped="",
            photo_urls=[],
        ),
        # Animal missing primary photo (should be filtered out)
        Animal(
            id=3,
            name="Charlie",
            breed="Beagle",
            age="Senior",
            sex="Male",
            location={},
            description="",
            url="http://example.com",
            primary_photo="",
            primary_photo_cropped="",
            photo_urls=[],
        ),
        # Animal missing URL (should be filtered out)
        Animal(
            id=4,
            name="Lucy",
            breed="Terrier",
            age="Young",
            sex="Female",
            location={},
            description="",
            url="",
            primary_photo="http://example.com/photo.jpg",
            primary_photo_cropped="http://example.com/photo_cropped.jpg",
            photo_urls=[],
        ),
    ]

    # Call the method
    sanitized = scraper.sanitize_animals(animals)

    # Verify the results
    assert len(sanitized) == 2  # Only 2 valid animals
    assert sanitized[0].id == 1
    assert sanitized[1].id == 2
    assert sanitized[1].primary_photo_cropped == sanitized[1].primary_photo  # Should be copied


@pytest.mark.asyncio
async def test_get_animal_signature():
    """Test creating unique signatures for animals"""
    scraper = PetfinderScraper()

    animal1 = Animal(
        id=1,
        name="Buddy",
        breed="Lab",
        age="",
        sex="",
        location={},
        description="Good dog",
        url="",
        primary_photo="",
        primary_photo_cropped="",
        photo_urls=[],
    )

    animal2 = Animal(
        id=2,
        name="Buddy",
        breed="Lab",
        age="",
        sex="",
        location={},
        description="Good dog",
        url="",
        primary_photo="",
        primary_photo_cropped="",
        photo_urls=[],
    )

    animal3 = Animal(
        id=3,
        name="Max",
        breed="Poodle",
        age="",
        sex="",
        location={},
        description="Nice dog",
        url="",
        primary_photo="",
        primary_photo_cropped="",
        photo_urls=[],
    )

    # Same animals should have same signature
    assert scraper.get_animal_signature(animal1) == scraper.get_animal_signature(animal2)
    # Different animals should have different signatures
    assert scraper.get_animal_signature(animal1) != scraper.get_animal_signature(animal3)


@pytest.mark.asyncio
async def test_save_progress(tmp_path):
    """Test saving progress to file"""
    output_file = os.path.join(tmp_path, "test_output.jsonl")

    scraper = PetfinderScraper()
    scraper.collected_pets = [
        Animal(
            id=1,
            name="Buddy",
            breed="Lab",
            age="Young",
            sex="Male",
            location={"city": "Brooklyn"},
            description="Good dog",
            url="http://example.com",
            primary_photo="http://example.com/photo.jpg",
            primary_photo_cropped="http://example.com/photo_cropped.jpg",
            photo_urls=[],
        ),
        # Duplicate animal (should be filtered)
        Animal(
            id=2,
            name="Buddy",
            breed="Lab",
            age="Young",
            sex="Male",
            location={"city": "Brooklyn"},
            description="Good dog",
            url="http://example.com",
            primary_photo="http://example.com/photo.jpg",
            primary_photo_cropped="http://example.com/photo_cropped.jpg",
            photo_urls=[],
        ),
        # Unique animal
        Animal(
            id=3,
            name="Max",
            breed="Poodle",
            age="Adult",
            sex="Male",
            location={"city": "Manhattan"},
            description="Nice dog",
            url="http://example.com",
            primary_photo="http://example.com/photo.jpg",
            primary_photo_cropped="http://example.com/photo_cropped.jpg",
            photo_urls=[],
        ),
    ]

    # Call the method
    scraper.save_progress(output_file)

    # Verify the file was created and contains the expected data
    assert os.path.exists(output_file)

    # Read the file and check contents
    with open(output_file) as f:
        lines = f.readlines()
        assert len(lines) == 2  # Should have 2 unique animals

        # Parse the JSON lines
        animals = [json.loads(line) for line in lines]
        assert animals[0]["name"] == "Buddy"
        assert animals[1]["name"] == "Max"

    # Check that collected_pets was cleared
    assert len(scraper.collected_pets) == 0
    # Check that total_pets was updated
    assert scraper.total_pets == 2


@pytest.mark.asyncio
@patch("doggelganger.scrape.PetfinderScraper.get_new_token")
@patch("doggelganger.scrape.PetfinderScraper.fetch_page")
@patch("doggelganger.scrape.PetfinderScraper.process_batch")
@patch("doggelganger.scrape.PetfinderScraper.save_progress")
async def test_scrape_all_pets(mock_save_progress, mock_process_batch, mock_fetch_page, mock_get_token, tmp_path):
    """Test the main scraping function"""
    output_file = os.path.join(tmp_path, "test_output.jsonl")

    # Setup mocks
    mock_get_token.return_value = None

    # First page response
    pagination_info = PaginationInfo(count_per_page=100, total_count=250, current_page=1, total_pages=3)
    mock_fetch_page.return_value = (pagination_info, [])

    # Batch processing returns some animals
    mock_animals = [
        Animal(
            id=1,
            name="Buddy",
            breed="Lab",
            age="Young",
            sex="Male",
            location={},
            description="",
            url="http://example.com",
            primary_photo="http://example.com/photo.jpg",
            primary_photo_cropped="http://example.com/photo_cropped.jpg",
            photo_urls=[],
        )
    ]
    mock_process_batch.return_value = mock_animals

    # Create scraper and run
    scraper = PetfinderScraper()
    await scraper.scrape_all_pets(output_file, save_interval=10)

    # Verify the token was fetched
    mock_get_token.assert_called_once()

    # Verify first page was fetched to get pagination info
    mock_fetch_page.assert_called_once()

    # Verify process_batch was called for each batch
    assert mock_process_batch.call_count == 1  # With batch_size=10, we need 1 batch for 3 pages

    # Verify save_progress was called
    mock_save_progress.assert_called()


@pytest.mark.asyncio
@patch("doggelganger.scrape.PetfinderScraper.get_new_token")
async def test_check_token_refresh(mock_get_token):
    """Test token refresh logic"""
    import time

    scraper = PetfinderScraper()

    # Case 1: No token
    scraper.token = None
    await scraper.check_token()
    mock_get_token.assert_called_once()
    mock_get_token.reset_mock()

    # Case 2: Token expired
    scraper.token = "test_token"
    scraper.token_timestamp = time.time() - 3700  # Expired (1 hour + buffer)
    await scraper.check_token()
    mock_get_token.assert_called_once()
    mock_get_token.reset_mock()

    # Case 3: Valid token
    scraper.token = "test_token"
    scraper.token_timestamp = time.time() - 1800  # Not expired (30 minutes)
    await scraper.check_token()
    mock_get_token.assert_not_called()
