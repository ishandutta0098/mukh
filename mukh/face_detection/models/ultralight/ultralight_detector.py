"""
Ultra-Light face detection model implementation.

This module implements the Ultra-Light-Fast-Generic-Face-Detector-1MB.
Adapted from: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

Original implementation by Linzaer
"""

import os
from typing import List

import cv2
import numpy as np
import torch

from ....core.model_hub import download_ultralight_model
from ....core.types import BoundingBox, FaceDetection
from ..base_detector import BaseFaceDetector
from .vision.ssd.config.fd_config import define_img_size
from .vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from .vision.ssd.mb_tiny_RFB_fd import (
    create_Mb_Tiny_RFB_fd,
    create_Mb_Tiny_RFB_fd_predictor,
)


class UltralightDetector(BaseFaceDetector):
    """Ultra-Light face detector implementation.

    A 1MB size face detector optimized for edge devices. Provides bounding
    box detection without landmarks.

    Attributes:
        device: PyTorch device (CPU/CUDA) for model execution
        input_size: Input resolution for the model
        net: Neural network model
        predictor: Detection predictor instance
        confidence_threshold: Minimum confidence for valid detections
    """

    def __init__(
        self,
        net_type: str = "RFB",
        input_size: int = 320,
        confidence_threshold: float = 0.9,
        candidate_size: int = 1500,
        weights_path: str = None,
        labels_path: str = None,
    ):
        """Initializes the Ultra-Light face detector.

        Args:
            net_type: Network architecture type ("RFB" or "slim")
            input_size: Input image size for the model
            confidence_threshold: Minimum confidence threshold for detections
            candidate_size: Maximum number of candidate detections
            weights_path: Optional custom path to model weights file
            labels_path: Optional custom path to class labels file
        """
        super().__init__(confidence_threshold)

        # Download models from Hugging Face if not provided
        if weights_path is None or labels_path is None:
            try:
                model_variant = f"{net_type}-{input_size}"
                weights_path, labels_path = download_ultralight_model(model_variant)
            except Exception as e:
                raise Exception(f"Failed to download UltraLight models: {str(e)}")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.candidate_size = candidate_size

        # Define image size before importing predictor
        define_img_size(input_size)

        # Load class names
        self.class_names = [name.strip() for name in open(labels_path).readlines()]

        # Initialize network based on type
        if net_type == "slim":
            self.net = create_mb_tiny_fd(
                len(self.class_names), is_test=True, device=self.device
            )
            self.predictor = create_mb_tiny_fd_predictor(
                self.net, candidate_size=candidate_size, device=self.device
            )
        elif net_type == "RFB":
            self.net = create_Mb_Tiny_RFB_fd(
                len(self.class_names), is_test=True, device=self.device
            )
            self.predictor = create_Mb_Tiny_RFB_fd_predictor(
                self.net, candidate_size=candidate_size, device=self.device
            )
        else:
            raise ValueError(f"Unsupported net_type: {net_type}")

        # Load model weights
        self.net.load(weights_path)

    def detect(
        self,
        image_path: str,
        save_csv: bool = False,
        csv_path: str = "detections.csv",
        save_annotated: bool = False,
        output_folder: str = "output",
    ) -> List[FaceDetection]:
        """Detects faces in the given image using Ultra-Light model.

        The image is resized to self.input_size for inference and results
        are scaled back to original image size.

        Args:
            image_path: Path to the input image.
            save_csv: Whether to save detection results to CSV file.
            csv_path: Path where to save the CSV file.
            save_annotated: Whether to save annotated image with bounding boxes.
            output_folder: Folder path where to save annotated images.

        Returns:
            List[FaceDetection]: List of detected faces, each containing:
                - bbox: BoundingBox with coordinates and confidence
                - landmarks: None (Ultralight doesn't provide landmarks)
        """
        # Load image from path
        image = self._load_image(image_path)
        orig_height, orig_width = image.shape[:2]

        # Resize image to input size
        resized_image = cv2.resize(image, (self.input_size, self.input_size))

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Get detections
        boxes, labels, probs = self.predictor.predict(
            image_rgb, self.candidate_size / 2, self.confidence_threshold
        )

        # Scale factors for converting back to original size
        width_scale = orig_width / self.input_size
        height_scale = orig_height / self.input_size

        # Convert to FaceDetection objects
        faces = []
        for i in range(boxes.size(0)):
            box = boxes[i, :].int().tolist()
            # Scale bounding box back to original image size
            bbox = BoundingBox(
                x1=int(box[0] * width_scale),
                y1=int(box[1] * height_scale),
                x2=int(box[2] * width_scale),
                y2=int(box[3] * height_scale),
                confidence=probs[i],
            )
            # Ultralight doesn't provide landmarks, so we pass None
            faces.append(FaceDetection(bbox=bbox))

        # Save to CSV if requested
        if save_csv:
            self._save_detections_to_csv(faces, image_path, csv_path)

        # Save annotated image if requested
        if save_annotated:
            self._save_annotated_image(image, faces, image_path, output_folder)

        return faces
