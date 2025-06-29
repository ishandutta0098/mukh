"""Simple tests for face_detector.py that don't require heavy dependencies"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest


# Mock all the heavy dependencies
def setup_mocks():
    """Setup all necessary mocks for testing"""
    # Mock torch and related modules
    torch_mock = MagicMock()
    torch_mock.nn = MagicMock()
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.__version__ = "2.0.0"

    # Mock mediapipe
    mp_mock = MagicMock()
    mp_mock.solutions = MagicMock()
    mp_mock.solutions.face_detection = MagicMock()

    # Apply all mocks
    modules_to_mock = [
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "torchvision",
        "torchvision.transforms",
        "mediapipe",
        "cv2",
        "face_alignment",
        "facenet_pytorch",
        "efficientnet_pytorch",
    ]

    for module in modules_to_mock:
        sys.modules[module] = torch_mock if module.startswith("torch") else mp_mock

    # Specific mocks
    sys.modules["torch"] = torch_mock
    sys.modules["mediapipe"] = mp_mock
    sys.modules["cv2"] = MagicMock()


# Setup mocks before importing anything from mukh
setup_mocks()

# Now safe to import
from mukh.face_detection.face_detector import DetectorType, FaceDetector


class TestFaceDetectorBasic:
    """Basic tests for the FaceDetector factory class that don't instantiate detectors."""

    def test_list_available_models_returns_correct_list(self):
        """Test that list_available_models returns the expected model names."""
        expected_models = ["blazeface", "mediapipe", "ultralight"]
        actual_models = FaceDetector.list_available_models()

        assert actual_models == expected_models
        assert len(actual_models) == 3

    def test_list_available_models_returns_list_type(self):
        """Test that list_available_models returns a list."""
        models = FaceDetector.list_available_models()
        assert isinstance(models, list)

    def test_list_available_models_contains_strings(self):
        """Test that all items in the returned list are strings."""
        models = FaceDetector.list_available_models()
        assert all(isinstance(model, str) for model in models)

    def test_create_invalid_detector_raises_value_error(self):
        """Test that creating an invalid detector raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            FaceDetector.create("invalid_model")

        assert "Unknown detector model: invalid_model" in str(exc_info.value)
        assert "Available models: ['blazeface', 'mediapipe', 'ultralight']" in str(
            exc_info.value
        )

    def test_create_none_detector_raises_value_error(self):
        """Test that passing None as model raises ValueError."""
        with pytest.raises(ValueError):
            FaceDetector.create(None)

    def test_create_empty_string_detector_raises_value_error(self):
        """Test that passing empty string as model raises ValueError."""
        with pytest.raises(ValueError):
            FaceDetector.create("")

    @pytest.mark.parametrize(
        "invalid_model",
        [
            "invalid",
            "BLAZEFACE",
            "MediaPipe",
            "ultra_light",
            "blazeface_v2",
            "yolo",
            "opencv",
            123,
        ],
    )
    def test_create_invalid_models_parametrized(self, invalid_model):
        """Parametrized test for various invalid model names."""
        with pytest.raises(ValueError):
            FaceDetector.create(invalid_model)

    def test_create_list_model_raises_type_error(self):
        """Test that passing a list as model raises TypeError."""
        with pytest.raises(TypeError):
            FaceDetector.create(["blazeface"])

    def test_error_message_format(self):
        """Test the specific format of error messages."""
        try:
            FaceDetector.create("unknown_model")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            assert "Unknown detector model: unknown_model" in error_msg
            assert "Available models:" in error_msg
            assert "blazeface" in error_msg
            assert "mediapipe" in error_msg
            assert "ultralight" in error_msg

    def test_consistency_between_create_and_list_methods(self):
        """Test that all models listed correspond to the create method's expectations."""
        available_models = FaceDetector.list_available_models()

        # Check that the internal detectors dict has the same keys
        # This tests the consistency without actually creating instances
        from mukh.face_detection.face_detector import FaceDetector as FD

        # We can access the detectors mapping through inspection if it was a public method
        # For now, we'll test that the error message shows the same models
        try:
            FD.create("nonexistent")
        except ValueError as e:
            error_msg = str(e)
            for model in available_models:
                assert model in error_msg


class TestFaceDetectorWithMocks:
    """Tests that use mocks for the detector classes."""

    @patch("mukh.face_detection.face_detector.BlazeFaceDetector")
    def test_create_blazeface_calls_constructor(self, mock_blazeface):
        """Test that creating BlazeFace detector calls its constructor."""
        mock_instance = Mock()
        mock_blazeface.return_value = mock_instance

        result = FaceDetector.create("blazeface")

        mock_blazeface.assert_called_once()
        assert result == mock_instance

    @patch("mukh.face_detection.face_detector.MediaPipeFaceDetector")
    def test_create_mediapipe_calls_constructor(self, mock_mediapipe):
        """Test that creating MediaPipe detector calls its constructor."""
        mock_instance = Mock()
        mock_mediapipe.return_value = mock_instance

        result = FaceDetector.create("mediapipe")

        mock_mediapipe.assert_called_once()
        assert result == mock_instance

    @patch("mukh.face_detection.face_detector.UltralightDetector")
    def test_create_ultralight_calls_constructor(self, mock_ultralight):
        """Test that creating Ultralight detector calls its constructor."""
        mock_instance = Mock()
        mock_ultralight.return_value = mock_instance

        result = FaceDetector.create("ultralight")

        mock_ultralight.assert_called_once()
        assert result == mock_instance

    @patch("mukh.face_detection.face_detector.BlazeFaceDetector")
    def test_multiple_detector_creation_independence(self, mock_blazeface):
        """Test that creating multiple detectors returns independent instances."""
        mock_instance1 = Mock()
        mock_instance2 = Mock()
        mock_blazeface.side_effect = [mock_instance1, mock_instance2]

        detector1 = FaceDetector.create("blazeface")
        detector2 = FaceDetector.create("blazeface")

        # Each call should return a new instance
        assert detector1 is not detector2
        assert mock_blazeface.call_count == 2

    def test_factory_methods_are_static(self):
        """Test that factory methods can be called without instantiation."""
        # Should work without creating a FaceDetector instance
        models = FaceDetector.list_available_models()

        assert models is not None
        assert len(models) > 0
