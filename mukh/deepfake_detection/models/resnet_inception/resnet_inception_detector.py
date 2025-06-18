"""ResNet Inception deepfake detection model implementation.

This module implements a deepfake detection model using InceptionResnetV1
with explainability features through GradCAM visualization.

GitHub: https://github.com/aaronespasa/deepfake-detection
LICENSE: Apache License 2.0
"""

import os
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from ....core.model_hub import download_resnet_inception_model
from ....core.types import DeepfakeDetection
from ..base import BaseDeepfakeDetector


class ResNetInceptionDetector(BaseDeepfakeDetector):
    """ResNet Inception deepfake detector implementation.

    A deepfake detection model using InceptionResnetV1 with explainability
    features through GradCAM visualization.

    Attributes:
        device: PyTorch device (CPU/CUDA) for model execution
        mtcnn: MTCNN face detector for preprocessing
        model: InceptionResnetV1 neural network model
        confidence_threshold: Minimum confidence for valid detections
    """

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.5,
        device: str = None,
    ):
        """Initializes the ResNet Inception deepfake detector.

        Args:
            model_path: Path to the trained model checkpoint. If None, downloads from Hugging Face.
            confidence_threshold: Minimum confidence threshold for detections
            device: Device to run inference on ('cpu' or 'cuda'). Auto-detected if None
        """
        super().__init__(confidence_threshold)

        # Set device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize MTCNN for face detection
        self.mtcnn = (
            MTCNN(select_largest=False, post_process=False, device=self.device)
            .to(self.device)
            .eval()
        )

        # Initialize the model
        self.model = InceptionResnetV1(
            pretrained="vggface2", classify=True, num_classes=1, device=self.device
        )

        # Load the trained weights

        if model_path is None:
            try:
                model_path = download_resnet_inception_model()
            except Exception as e:
                raise Exception(f"Failed to download ResNet Inception model: {str(e)}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def detect_image(
        self,
        image_path: str,
        save_csv: bool = False,
        csv_path: str = "deepfake_detections.csv",
        save_annotated: bool = False,
        output_folder: str = "output",
    ) -> DeepfakeDetection:
        """Detects deepfake in the given image.

        Args:
            image_path: Path to the input image
            save_csv: Whether to save detection results to CSV file
            csv_path: Path where to save the CSV file
            save_annotated: Whether to save annotated image with results
            output_folder: Folder path where to save annotated images

        Returns:
            DeepfakeDetection object containing detection results

        Raises:
            ValueError: If no face is detected in the image
        """
        # Load image
        image = self._load_image(image_path)
        input_image = Image.open(image_path).convert("RGB")

        # Detect face in the image
        face = self.mtcnn(input_image)
        if face is None:
            raise ValueError("No face detected in the image")

        # Prepare face tensor
        face = face.unsqueeze(0)  # add batch dimension
        face = F.interpolate(
            face, size=(256, 256), mode="bilinear", align_corners=False
        )

        # Convert face to numpy for visualization
        prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
        prev_face = prev_face.astype("uint8")

        # Normalize face for model input
        face = face.to(self.device)
        face = face.to(torch.float32)
        face = face / 255.0
        face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        # Make prediction
        with torch.no_grad():
            output = torch.sigmoid(self.model(face).squeeze(0))
            is_deepfake = output.item() >= self.confidence_threshold
            confidence = output.item() if is_deepfake else (1 - output.item())
            confidence = round(confidence, 2)

            detection = DeepfakeDetection(
                frame_number=0,  # Single image, so frame 0
                is_deepfake=is_deepfake,
                confidence=confidence,
                model_name="ResNetInception",
            )

        if save_annotated:
            self._save_annotated_image(image, detection, image_path, output_folder)

        # Save to CSV if requested
        if save_csv:
            self._save_detections_to_csv(detection, image_path, csv_path)

        return detection, is_deepfake

    def detect_video(
        self,
        video_path: str,
        save_csv: bool = False,
        csv_path: str = "deepfake_detections.csv",
        save_annotated: bool = False,
        output_folder: str = "output",
        num_frames: int = 11,
    ) -> List[DeepfakeDetection]:
        """Detects deepfake in the given video using equally spaced frames.

        Args:
            video_path: Path to the input video
            save_csv: Whether to save detection results to CSV file
            csv_path: Path where to save the CSV file
            save_annotated: Whether to save annotated video with results
            output_folder: Folder path where to save annotated videos
            num_frames: Number of equally spaced frames to analyze (default: 11)

        Returns:
            List of DeepfakeDetection objects for analyzed frames
        """
        # Extract equally spaced frames
        extracted_frames = self._extract_equally_spaced_frames(video_path, num_frames)

        detections = []

        for frame_number, frame in extracted_frames:
            try:
                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Detect face in the frame
                face = self.mtcnn(pil_image)
                if face is not None:
                    # Prepare face tensor
                    face = face.unsqueeze(0)
                    face = F.interpolate(
                        face, size=(256, 256), mode="bilinear", align_corners=False
                    )

                    # Normalize face for model input
                    face = face.to(self.device)
                    face = face.to(torch.float32)
                    face = face / 255.0

                    # Make prediction
                    with torch.no_grad():
                        output = torch.sigmoid(self.model(face).squeeze(0))
                        is_deepfake = output.item() >= self.confidence_threshold
                        confidence = (
                            output.item() if is_deepfake else (1 - output.item())
                        )
                        confidence = round(confidence, 2)

                        detection = DeepfakeDetection(
                            frame_number=frame_number,
                            is_deepfake=is_deepfake,
                            confidence=confidence,
                            model_name="ResNetInception",
                        )

                        detections.append(detection)

            except Exception as e:
                # Skip frames with errors (e.g., no face detected)
                pass

        # Aggregate results and print final decision
        if detections:
            final_result, deepfake_count, total_frames = (
                self.aggregate_video_detections(
                    detections, video_path, output_folder, "ResNetInception"
                )
            )

        # Save annotated video if requested
        if save_annotated and detections:
            self._save_annotated_video(video_path, detections, output_folder)

        # Save to CSV if requested
        if save_csv and detections:
            self._save_detections_to_csv(detections, video_path, csv_path)
            # Save final aggregated result to text file
            self._save_final_video_result_to_txt(
                final_result,
                video_path,
                output_folder,
                "ResNetInception",
                deepfake_count,
                total_frames,
            )

        return detections, final_result
