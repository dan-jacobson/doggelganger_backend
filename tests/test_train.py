import numpy as np
import pytest

from doggelganger.train import make_training_data


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


def test_make_training_data(mock_data_dir, mocker):
    mock_get_embedding = mocker.patch("doggelganger.utils.get_embedding")
    mock_get_embedding.side_effect = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6]),
        np.array([0.7, 0.8, 0.9]),
        np.array([1.0, 1.1, 1.2]),
    ]

    X, y = make_training_data(mock_data_dir)

    assert len(X) == 2
    assert len(y) == 2
    assert all(isinstance(emb, np.ndarray) for emb in X)
    assert all(isinstance(emb, np.ndarray) for emb in y)


