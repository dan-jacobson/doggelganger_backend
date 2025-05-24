import asyncio
from dataclasses import asdict
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import jsonlines
import pytest
from aioresponses import aioresponses
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
        primary_photo_cropped="https://example.com/buddy_cropped.jpg",
        photo_urls=None,
    )


@pytest.fixture
def sample_animals(sample_animal):
    animals = []
    for i in range(5):
        animal = Animal(
            id=f"12{i}",
            name=f"Buddy{i}",
            breed="Labrador",
            age="Young",
            sex="Male",
            location={"city": "Brooklyn", "state": "NY", "postcode": ""},
            description="He's a real good boy.",
            url=f"https://example.com/buddy{i}",
            primary_photo=f"https://example.com/buddy{i}.jpg",
            primary_photo_cropped=f"https://example.com/buddy{i}_cropped.jpg",
            photo_urls=None,
        )
        animals.append(animal)
    return animals


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
        test_animals = sample_animals[:2]
        
        with aioresponses() as m:
            # Mock all the URLs that will be requested
            for animal in test_animals:
                url = getattr(animal, "primary_photo_cropped")
                m.get(url, body=mock_image, status=200)
            
            dataset = AsyncDogDataset(metadata=test_animals, show_progress=False)
            
            producer_task = asyncio.create_task(dataset.producer())
            
            items = []
            try:
                # Get expected number of items
                for _ in range(len(test_animals)):
                    item = await asyncio.wait_for(dataset.image_queue.get(), timeout=5.0)
                    if item is None:  # Stop signal
                        break
                    items.append(item)
                
                # Get stop signal
                stop_signal = await asyncio.wait_for(dataset.image_queue.get(), timeout=5.0)
                assert stop_signal is None
                
            except asyncio.TimeoutError:
                pytest.fail("Producer test timed out")
            finally:
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
            
            assert len(items) == len(test_animals)
            assert all(isinstance(item[0], Animal) for item in items)
            assert all(isinstance(item[1], Image.Image) for item in items)

    @pytest.mark.asyncio
    async def test_consumer(self, sample_animals, mock_model):
        """Test the consumer processes items from the queue correctly"""
        test_animals = sample_animals[:3]
        dataset = AsyncDogDataset(metadata=test_animals, batch_size=2, show_progress=False)

        # Pre-populate queue with test data
        for animal in test_animals:
            image = Image.new("RGB", (100, 100), color="red")
            await dataset.image_queue.put((animal, image))
        
        # Add stop signal
        await dataset.image_queue.put(None)

        mock_db = MagicMock()

        # Run consumer with timeout
        try:
            await asyncio.wait_for(dataset.consumer(mock_model, mock_db), timeout=10.0)
        except asyncio.TimeoutError:
            pytest.fail("Consumer test timed out")

        # Verify results - should have 2 batches (2 items + 1 item)
        assert mock_db.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_process_all(self, sample_animals, mock_model, mock_image):
        """Test the end-to-end processing flow"""
        test_animals = sample_animals[:2]  # Keep it small for testing
        
        with aioresponses() as m:
            # Mock all the URLs that will be requested
            for animal in test_animals:
                url = getattr(animal, "primary_photo_cropped")
                m.get(url, body=mock_image, status=200)
            
            dataset = AsyncDogDataset(metadata=test_animals, batch_size=2, show_progress=False)
            mock_db = MagicMock()

            try:
                records = await asyncio.wait_for(
                    dataset.process_all(mock_model, mock_db),
                    timeout=15.0
                )
                assert len(records) <= len(test_animals)
            except asyncio.TimeoutError:
                pytest.fail('process_all test timed out')

            # Check results
            assert all(isinstance(record, Record) for record in records)

            # Check that upsert was called
            assert mock_db.upsert.call_count > 0

    def test_generate_records(self, sample_animals, mock_model):
        """Test record generation from images and metadata"""
        # Create some test images
        images = [Image.new("RGB", (100, 100), color="red") for _ in sample_animals]
        dataset = AsyncDogDataset(metadata=sample_animals, show_progress=False)

        records = dataset.generate_records(mock_model, images, sample_animals)

        assert len(records) == len(sample_animals)
        assert all(isinstance(record, Record) for record in records)
        assert all(record.id == str(animal.id) for record, animal in zip(records, sample_animals, strict=False))
        assert all(len(record.embedding) == 512 for record in records)
        # make sure we have the non-ID keys from metadata in each record
        assert all(key in record.metadata for record in records for key in asdict(sample_animals[0]) if key != "id")


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
