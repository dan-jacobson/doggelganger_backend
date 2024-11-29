import numpy as np
import pytest

from doggelganger.train import (
    align_animal_to_human_embeddings,
    align_embedding,
    make_embeddings,
)


@pytest.fixture
def mock_data_dir(tmp_path):
    human_dir = tmp_path / "human"
    animal_dir = tmp_path / "animal"
    human_dir.mkdir()
    animal_dir.mkdir()

    # Create mock image files
    (human_dir / "image1.jpg").touch()
    (human_dir / "image2.jpg").touch()
    (animal_dir / "image1.jpg").touch()
    (animal_dir / "image2.jpg").touch()

    return tmp_path


@pytest.fixture
def mock_embeddings():
    return {
        "human": {
            "image1.jpg": np.array([0.1, 0.2, 0.3]),
            "image2.jpg": np.array([0.4, 0.5, 0.6]),
        },
        "animal": {
            "image1.jpg": np.array([0.7, 0.8, 0.9]),
            "image2.jpg": np.array([1.0, 1.1, 1.2]),
        },
    }


def test_make_embeddings(mock_data_dir, mocker):
    mock_get_embedding = mocker.patch("doggelganger.utils.get_embedding")
    mock_get_embedding.side_effect = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6]),
        np.array([0.7, 0.8, 0.9]),
        np.array([1.0, 1.1, 1.2]),
    ]

    human_embeddings, animal_embeddings = make_embeddings(mock_data_dir)

    assert len(human_embeddings) == 2
    assert len(animal_embeddings) == 2
    assert all(isinstance(emb, np.ndarray) for emb in human_embeddings.values())
    assert all(isinstance(emb, np.ndarray) for emb in animal_embeddings.values())


def test_align_animal_to_human_embeddings(mock_embeddings):
    human_embeddings = mock_embeddings["human"]
    animal_embeddings = mock_embeddings["animal"]

    model, X, y = align_animal_to_human_embeddings(human_embeddings, animal_embeddings)

    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert X.shape == (2, 3)
    assert y.shape == (2, 3)


def test_align_embedding():
    embedding = np.array([0.1, 0.2, 0.3])
    coef = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    intercept = np.array([0.1, 0.2, 0.3])

    aligned_embedding = align_embedding(embedding, coef, intercept)

    assert aligned_embedding.shape == (3,)
    assert np.allclose(aligned_embedding, np.array([1.4, 3.2, 5.0]))
