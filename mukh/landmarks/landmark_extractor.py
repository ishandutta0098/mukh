"""Landmark extraction module providing a unified interface for multiple extraction models.

This module provides a factory class for creating landmark extractors with different
underlying implementations. It supports multiple extraction models through a consistent
interface.

Example:
    Basic usage with default settings:

    >>> from mukh.landmarks import LandmarkExtractor
    >>> extractor = LandmarkExtractor.create("blazeface")
    >>> landmarks = extractor.extract("image.jpg")

    List available models:

    >>> LandmarkExtractor.list_available_models()
    ['blazeface', 'mediapipe']
"""

from typing import List, Literal

from .models.base_extractor import BaseLandmarkExtractor
from .models.blazeface import BlazeFaceLandmarkExtractor

ExtractorType = Literal["blazeface", "mediapipe"]


class LandmarkExtractor:
    """Factory class for creating landmark extraction model instances.

    This class provides a unified interface to create and use different landmark extraction
    models through a consistent API.
    """

    @staticmethod
    def create(model: ExtractorType) -> BaseLandmarkExtractor:
        """Creates a landmark extractor instance of the specified type.

        Args:
            model: The type of extractor to create. Must be one of: "blazeface" or
                "mediapipe".

        Returns:
            A BaseLandmarkExtractor instance of the requested type.

        Raises:
            ValueError: If the specified model type is not supported.
        """
        detectors = {
            "blazeface": BlazeFaceLandmarkExtractor,
        }

        if model not in detectors:
            raise ValueError(
                f"Unknown detector model: {model}. "
                f"Available models: {list(detectors.keys())}"
            )

        return detectors[model]()

    @staticmethod
    def list_available_models() -> List[str]:
        """Returns a list of available face detection model names.

        Returns:
            List of strings containing supported model names.
        """
        return ["blazeface", "mediapipe"]
