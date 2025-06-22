"""BlazeFace face detection model implementation.

This module implements the BlazeFace face detection model from MediaPipe.
Adapted from: https://github.com/hollance/BlazeFace-PyTorch
Original implementation by M.I. Hollemans.

The model is optimized for mobile devices and provides both bounding box
detection and 6 facial landmarks.

LICENSE: Apache License 2.0
"""

from typing import List

import cv2
import numpy as np
import torch

from ....core.model_hub import download_blazeface_models
from ....core.types import BoundingBox, FaceDetection
from ..base_detector import BaseFaceDetector
from .blazeface_torch import BlazeFace


class BlazeFaceDetector(BaseFaceDetector):
    """BlazeFace face detector implementation.

    A lightweight face detector that provides both bounding boxes and facial
    landmarks. Optimized for mobile devices.

    Attributes:
        device: PyTorch device (CPU/CUDA) for model execution
        net: BlazeFace neural network model
        confidence_threshold: Minimum confidence for valid detections
    """

    def __init__(
        self,
        weights_path: str = None,
        anchors_path: str = None,
        confidence_threshold: float = 0.75,
        device: str = "cpu",
    ):
        """Initializes the BlazeFace detector.

        Args:
            weights_path: Optional custom path to model weights file
            anchors_path: Optional custom path to anchor boxes file
            confidence_threshold: Minimum confidence threshold for detections
            device: Device to run inference on ('cpu' or 'cuda')
        """
        super().__init__(confidence_threshold)

        # Download models from Hugging Face if not provided
        if weights_path is None or anchors_path is None:
            try:
                weights_path, anchors_path = download_blazeface_models()
            except Exception as e:
                raise Exception(f"Failed to download BlazeFace models: {str(e)}")

        self.device = torch.device(device)
        self.net = BlazeFace().to(self.device)
        self.net.load_weights(weights_path)
        self.net.load_anchors(anchors_path)

        # Set minimum score threshold
        self.net.min_score_thresh = confidence_threshold

    def detect(
        self,
        image_path: str,
        save_json: bool = True,
        json_path: str = "detections.json",
        save_annotated: bool = False,
        output_folder: str = "output",
    ) -> List[FaceDetection]:
        """Detects faces in the given image.

        The image is resized to 128x128 pixels for inference and the results
        are scaled back to the original image size.

        Args:
            image_path: Path to the input image.
            save_json: Whether to save detection results to JSON file, defaults to True.
            json_path: Path where to save the JSON file.
            save_annotated: Whether to save annotated image with bounding boxes.
            output_folder: Folder path where to save annotated images.

        Returns:
            List[FaceDetection]: List of detected faces, each containing:
                - bbox: BoundingBox with coordinates and confidence
                - landmarks: Array of 6 facial keypoints
        """
        # Load image from path
        image = self._load_image(image_path)

        # Get original dimensions
        orig_h, orig_w = image.shape[:2]

        # Resize to 128x128
        image_resized = cv2.resize(image, (128, 128))

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Get detections
        detections = self.net.predict_on_image(image_rgb)

        # Apply NMS to filter overlapping detections
        if len(detections) > 0:
            # Convert to batch format for NMS
            detections_batch = [detections]
            filtered_detections = self.net.nms(detections_batch)
            detections = filtered_detections[0]

        # Convert to FaceDetection objects
        faces = []
        for detection in detections:
            # Convert normalized coordinates back to original image size
            x1 = float(detection[1]) * orig_w  # xmin
            y1 = float(detection[0]) * orig_h  # ymin
            x2 = float(detection[3]) * orig_w  # xmax
            y2 = float(detection[2]) * orig_h  # ymax

            bbox = BoundingBox(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                confidence=detection[16].item(),
            )

            faces.append(FaceDetection(bbox=bbox))

        # Save to JSON if requested
        if save_json:
            self._save_detections_to_json(faces, image_path, json_path)

        # Save annotated image if requested
        if save_annotated:
            self._save_annotated_image(image, faces, image_path, output_folder)

        return faces
