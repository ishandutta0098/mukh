"""MediaPipe face detection model implementation.

This module implements Google's MediaPipe face detection model.
Source: https://github.com/google-ai-edge/mediapipe

LICENSE: Apache License 2.0
"""

from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from ....core.types import BoundingBox, FaceDetection
from ..base_detector import BaseFaceDetector


class MediaPipeFaceDetector(BaseFaceDetector):
    """MediaPipe face detector implementation.

    Uses Google's MediaPipe framework for real-time face detection with
    landmarks.

    Attributes:
        mp_face_detection: MediaPipe face detection solution
        face_detection: Configured face detector instance
        confidence_threshold: Minimum confidence for valid detections
    """

    def __init__(self, confidence_threshold: float = 0.5, model_selection: int = 0):
        """Initializes the MediaPipe face detector.

        Args:
            confidence_threshold: Minimum confidence threshold for detections
            model_selection: Model type selection (0=default, 1=full range)
        """
        super().__init__(confidence_threshold)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=confidence_threshold,
            model_selection=model_selection,
        )

    def detect(
        self,
        image_path: str,
        save_csv: bool = False,
        csv_path: str = "detections.csv",
        save_annotated: bool = False,
        output_folder: str = "output",
    ) -> List[FaceDetection]:
        """Detects faces in the given image using MediaPipe.

        Args:
            image_path: Path to the input image.
            save_csv: Whether to save detection results to CSV file.
            csv_path: Path where to save the CSV file.
            save_annotated: Whether to save annotated image with bounding boxes.
            output_folder: Folder path where to save annotated images.

        Returns:
            List[FaceDetection]: List of detected faces, each containing:
                - bbox: BoundingBox with coordinates and confidence
                - landmarks: Array of 6 facial keypoints
        """
        # Load image from path
        image = self._load_image(image_path)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        results = self.face_detection.process(image_rgb)

        faces = []
        if results.detections:
            image_height, image_width, _ = image.shape
            for detection in results.detections:
                bbox_rel = detection.location_data.relative_bounding_box

                # Convert relative coordinates to absolute pixels
                x = int(bbox_rel.xmin * image_width)
                y = int(bbox_rel.ymin * image_height)
                w = int(bbox_rel.width * image_width)
                h = int(bbox_rel.height * image_height)

                bbox = BoundingBox(
                    x1=x, y1=y, x2=x + w, y2=y + h, confidence=detection.score[0]
                )

                faces.append(FaceDetection(bbox=bbox))

        # Save to CSV if requested
        if save_csv:
            self._save_detections_to_csv(faces, image_path, csv_path)

        # Save annotated image if requested
        if save_annotated:
            self._save_annotated_image(image, faces, image_path, output_folder)

        return faces
