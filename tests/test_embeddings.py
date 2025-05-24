import asyncio
from dataclasses import asdict
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import jsonlines
import pytest
from PIL import Image

from doggelganger.embeddings import AsyncDogDataset, Record, load_metadata, process_dogs
from doggelganger.utils import Animal


@pytest.fixture
def sample_animal():
    return Animal(
        id="123",
        name="Buddy",
        breed="Labrador",
        age="Young",
        sex="Male",
        location={"city": "Brooklyn", "state": "NY", "postcode": ""},
        description="He's a real good boy.",
        url="https://example.com/buddy",
        primary_photo="https://example.com/buddy.jpg",
        primary_photo_cropped="https://example.com/buddy.jpg",
        photo_urls=None,
    )


@pytest.fixture
def sample_animals(sample_animal):
    return [sample_animal] * 5


@pytest.fixture
def sample_jsonl_path(tmp_path, sample_animals):
    """Create a temporary JSONL file with sample animal data"""
    file_path = tmp_path / "animals.jsonl"
    with jsonlines.open(file_path, mode="w") as writer:
        for animal in sample_animals:
            writer.write(animal.__dict__)
    return file_path


@pytest.fixture
def mock_image():
    """Create a small test image"""
    img = Image.new("RGB", (100, 100), color="red")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def mock_model():
    """Create a mock CLIP model"""
    model = MagicMock()
    # Mock the embedding generation to return a list of [embedding] for each image
    model.side_effect = lambda images, batch_size: [[[0.1] * 512] for _ in range(len(images))]
    return model


class TestRecord:
    def test_record_creation(self):
        """Test that Record objects can be created correctly"""
        record = Record(id="123", embedding=[0.1, 0.2], metadata={"name": "Buddy"})
        assert record.id == "123"
        assert record.embedding == [0.1, 0.2]
        assert record.metadata == {"name": "Buddy"}


class TestMetadataLoading:
    def test_load_metadata(self, sample_jsonl_path, sample_animals):
        """Test loading metadata from a JSONL file"""
        animals = load_metadata(sample_jsonl_path)
        assert len(animals) == len(sample_animals)
        assert isinstance(animals[0], Animal)
        assert animals[0].name == sample_animals[0].name


class TestAsyncDogDataset:
    @pytest.mark.asyncio
    async def test_producer(self, sample_animals, mock_image):
        """Test the producer function fills the queue correctly"""
        # Create a mock session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = mock_image
        mock_session.get.return_value.__aenter__.return_value = mock_response

        # Create the dataset with our sample animals
        dataset = AsyncDogDataset(metadata=sample_animals)

        # Replace aiohttp.ClientSession with our mock
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Start the producer
            producer_task = asyncio.create_task(dataset.producer())

            # Get items from the queue
            items = []
            for _ in range(len(sample_animals)):
                item = await dataset.image_queue.get()
                items.append(item)

            # Get the stop signal
            stop_signal = await dataset.image_queue.get()

            # Wait for the producer to finish
            await producer_task

            # Check results
            assert len(items) == len(sample_animals)
            assert all(isinstance(item[0], Animal) for item in items)
            assert all(isinstance(item[1], Image.Image) for item in items)
            assert stop_signal is None

    def test_generate_records(self, sample_animals, mock_model):
        """Test record generation from images and metadata"""
        # Create some test images
        images = [Image.new("RGB", (100, 100), color="red") for _ in sample_animals]

        dataset = AsyncDogDataset(metadata=sample_animals)
        records = dataset.generate_records(mock_model, images, sample_animals)

        assert len(records) == len(sample_animals)
        assert all(isinstance(record, Record) for record in records)
        assert all(record.id == str(animal.id) for record, animal in zip(records, sample_animals, strict=False))
        assert all(len(record.embedding) == 512 for record in records)
        # make sure we have the non-ID keys from metadata in each record
        assert all(key in record.metadata for record in records for key in asdict(sample_animals[0]) if key != "id")

    @pytest.mark.asyncio
    async def test_consumer(self, sample_animals, mock_model):
        """Test the consumer processes items from the queue correctly"""
        # Create the dataset
        dataset = AsyncDogDataset(metadata=sample_animals, batch_size=2)

        # Create some test images and put them in the queue
        for animal in sample_animals:
            image = Image.new("RGB", (100, 100), color="red")
            await dataset.image_queue.put((animal, image))

        # Add the stop signal
        await dataset.image_queue.put(None)

        # Mock the database
        mock_db = MagicMock()

        # Start the consumer
        await dataset.consumer(mock_model, mock_db)

        # Check that records were generated and added to the processed queue
        assert dataset.processed_queue.qsize() == 3  # 2 batches of 2 + 1 batch of 1 + stop signal

        # Check that upsert was called for each batch
        assert mock_db.upsert.call_count == 3

    @pytest.mark.asyncio
    async def test_process_all(self, sample_animals, mock_model, mock_image):
        """Test the end-to-end processing flow"""
        # Create a mock session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = mock_image
        mock_session.get.return_value.__aenter__.return_value = mock_response

        # Create the dataset
        dataset = AsyncDogDataset(metadata=sample_animals, batch_size=2)

        # Mock the database
        mock_db = MagicMock()

        # Replace aiohttp.ClientSession with our mock
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Process all animals
            records = await dataset.process_all(mock_model, mock_db)

            # Check results
            assert len(records) == len(sample_animals)
            assert all(isinstance(record, Record) for record in records)

            # Check that upsert was called
            assert mock_db.upsert.call_count > 0


@patch("doggelganger.embeddings.load_model")
@patch("doggelganger.embeddings.vecs")
class TestProcessDogs:
    def test_process_dogs_smoke_test(self, mock_vecs, mock_load_model, sample_jsonl_path, mock_model):
        """Test process_dogs with smoke_test flag"""
        mock_load_model.return_value = mock_model

        process_dogs(sample_jsonl_path, smoke_test=True)

        # Verify vecs was not called
        mock_vecs.create_client.assert_not_called()

        # Verify model was loaded
        mock_load_model.assert_called_once()

    def test_process_dogs_with_drop(self, mock_vecs, mock_load_model, sample_jsonl_path, mock_model):
        """Test process_dogs with drop_existing flag"""
        mock_load_model.return_value = mock_model
        mock_client = MagicMock()
        mock_vecs.create_client.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.index = False

        process_dogs(sample_jsonl_path, drop_existing=True)

        # Verify collection was deleted and recreated
        mock_client.delete_collection.assert_called_once()
        mock_client.get_or_create_collection.assert_called_once()
        mock_collection.create_index.assert_called_once()

    def test_process_dogs_with_limit(self, mock_vecs, mock_load_model, sample_jsonl_path, mock_model):
        """Test process_dogs with N limit"""
        mock_load_model.return_value = mock_model
        mock_client = MagicMock()
        mock_vecs.create_client.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.index = True

        process_dogs(sample_jsonl_path, N=2)

        # Verify collection was created but not deleted
        mock_client.delete_collection.assert_not_called()
        mock_client.get_or_create_collection.assert_called_once()
        # Index already exists, so create_index should not be called
        mock_collection.create_index.assert_not_called()
