"""Base class defining the interface for face detection implementations.

This module provides the abstract base class that all face detector implementations
must inherit from, ensuring a consistent interface across different models.
"""

import csv
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import cv2
import numpy as np
import json

from ...core.types import FaceDetection


class BaseFaceDetector(ABC):
    """Abstract base class for face detector implementations.

    All face detector implementations must inherit from this class and implement
    the required abstract methods.

    Attributes:
        confidence_threshold: Float threshold (0-1) for detection confidence.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """Initializes the face detector.

        Args:
            confidence_threshold: Minimum confidence threshold for detections.
                Defaults to 0.5.
        """
        self.confidence_threshold = confidence_threshold

    def _load_image(self, image_path: str) -> np.ndarray:
        """Loads an image from disk in BGR format.

        Args:
            image_path: Path to the image file.

        Returns:
            np.ndarray: The loaded image in BGR format.

        Raises:
            ValueError: If the image cannot be loaded from the given path.
        """
        if not os.path.exists(image_path):
            raise ValueError(f"Image path does not exist: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from: {image_path}")

        return image

    @abstractmethod
    def detect(
        self,
        image_path: str,
        save_json: bool = True,
        json_path: str = "detections.json",
        save_annotated: bool = False,
        output_folder: str = "output",
    ) -> List[FaceDetection]:
        """Detects faces in the given image.

        Args:
            image_path: Path to the input image.
            save_json: Whether to save detection results to JSON file, defaults to True.
            json_path: Path where to save the JSON file.
            save_annotated: Whether to save annotated image with bounding boxes.
            output_folder: Folder path where to save annotated images.

        Returns:
            List of FaceDetection objects containing detected faces.
        """
        pass

    def _save_detections_to_json(
        self, detections: List[FaceDetection], image_path: str, json_path: str
    ) -> None:
        """Saves face detection results to a JSON file.

        Args:
            detections: List of face detections to save.
            image_path: Path to the source image.
            json_path: Path where to save the JSON file.
        """
        # Extract just the filename from the full path
        image_name = os.path.basename(image_path)

        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(json_path) if os.path.dirname(json_path) else ".",
            exist_ok=True,
        )

        # Prepare data for JSON
        detection_results = []
        for detection in detections:
            bbox = detection.bbox
            detection_results.append({
                "image_name": image_name,
                "x1": bbox.x1,
                "y1": bbox.y1,
                "x2": bbox.x2,
                "y2": bbox.y2,
                "confidence": bbox.confidence,
            })

        # Write to JSON (overwrite or append logic can be modified as needed)
        with open(json_path, "w", encoding="utf-8") as jsonfile:
            json.dump(detection_results, jsonfile, indent=4)

    def _draw_detections(
        self, image: np.ndarray, faces: List[FaceDetection]
    ) -> np.ndarray:
        """Draws detection results on the image.

        Args:
            image: Input image as numpy array
            faces: List of detected faces

        Returns:
            np.ndarray: Copy of input image with bounding boxes and landmarks drawn
        """
        image_copy = image.copy()
        for face in faces:
            bbox = face.bbox
            # Draw bounding box
            cv2.rectangle(
                image_copy,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (0, 255, 0),
                2,
            )

            # Draw landmarks if available
            if face.landmarks is not None:
                for x, y in face.landmarks:
                    cv2.circle(image_copy, (int(x), int(y)), 2, (0, 255, 0), 2)

            # Add confidence score
            label = f"{bbox.confidence:.2f}"
            cv2.putText(
                image_copy,
                label,
                (int(bbox.x1), int(bbox.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return image_copy

    def _save_annotated_image(
        self,
        image: np.ndarray,
        faces: List[FaceDetection],
        image_path: str,
        output_folder: str,
    ) -> str:
        """Saves annotated image with detection results.

        Args:
            image: Original image
            faces: List of detected faces
            image_path: Path to the original image
            output_folder: Folder where to save the annotated image

        Returns:
            str: Path to the saved annotated image
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Draw detections on image
        annotated_image = self._draw_detections(image, faces)

        # Create output filename
        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        output_filename = f"{name}_detected{ext}"
        output_path = os.path.join(output_folder, output_filename)

        # Save annotated image
        cv2.imwrite(output_path, annotated_image)

        return output_path
