"""Global test configuration and fixtures for mukh tests."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


# Mock heavy dependencies before any imports
def mock_torch():
    torch_mock = MagicMock()
    torch_mock.nn = MagicMock()
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.__version__ = "2.0.0"
    return torch_mock


def mock_mediapipe():
    mp_mock = MagicMock()
    mp_mock.solutions = MagicMock()
    mp_mock.solutions.face_detection = MagicMock()
    return mp_mock


# Apply mocks
sys.modules["torch"] = mock_torch()
sys.modules["torch.nn"] = sys.modules["torch"].nn
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["torchvision"] = MagicMock()
sys.modules["torchvision.transforms"] = MagicMock()
sys.modules["mediapipe"] = mock_mediapipe()
sys.modules["cv2"] = MagicMock()

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def test_assets_dir() -> Path:
    """Provide path to test assets directory."""
    return PROJECT_ROOT / "assets"


@pytest.fixture(scope="session")
def test_images_dir(test_assets_dir: Path) -> Path:
    """Provide path to test images directory."""
    return test_assets_dir / "images"


@pytest.fixture(scope="session")
def test_videos_dir(test_assets_dir: Path) -> Path:
    """Provide path to test videos directory."""
    return test_assets_dir / "videos"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample RGB image for testing."""
    # Create a 224x224x3 RGB image with a simple pattern
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    # Add some patterns to make it more realistic
    image[:112, :112] = [255, 0, 0]  # Red square
    image[:112, 112:] = [0, 255, 0]  # Green square
    image[112:, :112] = [0, 0, 255]  # Blue square
    image[112:, 112:] = [255, 255, 0]  # Yellow square

    return image


@pytest.fixture
def sample_image_with_face() -> np.ndarray:
    """Create a sample image with a face-like pattern for testing."""
    # Create a 224x224x3 RGB image
    image = np.ones((224, 224, 3), dtype=np.uint8) * 128  # Gray background

    # Add a simple face-like pattern
    # Face oval
    center_x, center_y = 112, 112
    for y in range(224):
        for x in range(224):
            # Simple oval shape
            if ((x - center_x) / 60) ** 2 + ((y - center_y) / 80) ** 2 <= 1:
                image[y, x] = [220, 180, 140]  # Skin color

    # Eyes
    image[90:100, 90:100] = [0, 0, 0]  # Left eye
    image[90:100, 124:134] = [0, 0, 0]  # Right eye

    # Mouth
    image[140:150, 102:122] = [200, 100, 100]  # Mouth

    return image


@pytest.fixture
def sample_image_path(temp_dir: Path, sample_image: np.ndarray) -> Path:
    """Create a temporary image file and return its path."""
    image_path = temp_dir / "sample_image.jpg"

    # Convert numpy array to PIL Image and save
    pil_image = Image.fromarray(sample_image)
    pil_image.save(image_path)

    return image_path


@pytest.fixture
def sample_batch_images() -> np.ndarray:
    """Create a batch of sample images for testing."""
    batch_size = 4
    batch = np.zeros((batch_size, 224, 224, 3), dtype=np.uint8)

    for i in range(batch_size):
        # Create different patterns for each image
        color_value = (i + 1) * 60
        batch[i] = np.full((224, 224, 3), color_value, dtype=np.uint8)

        # Add some variation
        batch[i, i * 50 : (i + 1) * 50, i * 50 : (i + 1) * 50] = [255, 255, 255]

    return batch


@pytest.fixture
def mock_face_detection_result():
    """Create a mock face detection result for testing."""
    from mukh.core.types import BoundingBox, FaceDetection

    bbox = BoundingBox(x=50, y=50, width=100, height=120)

    return FaceDetection(
        bbox=bbox,
        confidence=0.95,
        landmarks=np.array(
            [
                [75, 75],  # Left eye
                [125, 75],  # Right eye
                [100, 90],  # Nose
                [85, 110],  # Left mouth corner
                [115, 110],  # Right mouth corner
            ]
        ),
    )


@pytest.fixture
def mock_multiple_face_detections():
    """Create multiple mock face detection results for testing."""
    from mukh.core.types import BoundingBox, FaceDetection

    detections = []

    # First face
    bbox1 = BoundingBox(x=50, y=50, width=100, height=120)
    detection1 = FaceDetection(
        bbox=bbox1,
        confidence=0.95,
        landmarks=np.array([[75, 75], [125, 75], [100, 90], [85, 110], [115, 110]]),
    )
    detections.append(detection1)

    # Second face
    bbox2 = BoundingBox(x=200, y=80, width=80, height=100)
    detection2 = FaceDetection(
        bbox=bbox2,
        confidence=0.88,
        landmarks=np.array(
            [[220, 100], [260, 100], [240, 115], [225, 135], [255, 135]]
        ),
    )
    detections.append(detection2)

    return detections


# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that test individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for multiple components"
    )
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")
    config.addinivalue_line("markers", "gpu: Tests that require GPU support")
    config.addinivalue_line("markers", "network: Tests that require network access")


# Skip tests based on conditions
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on configuration."""
    # Skip GPU tests if no GPU is available
    skip_gpu = pytest.mark.skip(reason="GPU not available")

    for item in items:
        if "gpu" in item.keywords:
            # Check if GPU is available (simplified check)
            try:
                import torch

                if not torch.cuda.is_available():
                    item.add_marker(skip_gpu)
            except ImportError:
                item.add_marker(skip_gpu)

        # Skip network tests if running in offline mode
        if "network" in item.keywords and config.getoption("--offline", default=False):
            skip_network = pytest.mark.skip(reason="Running in offline mode")
            item.add_marker(skip_network)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--offline",
        action="store_true",
        default=False,
        help="Run tests in offline mode (skip network tests)",
    )

    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests that are normally skipped",
    )
