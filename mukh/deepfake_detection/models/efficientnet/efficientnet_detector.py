"""EfficientNet deepfake detection model implementation.

This module implements a deepfake detection model using EfficientNet architecture
based on the fornet library implementation.

Author Credits:

GitHub: https://github.com/polimi-ispl/icpr2020dfdc
License: GNU GENERAL PUBLIC LICENSE 3.0
"""

from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.model_zoo import load_url
from torchvision import transforms

from mukh.deepfake_detection.models.efficientnet.architectures import fornet, weights

from ....core.types import DeepfakeDetection
from ..base import BaseDeepfakeDetector


class EfficientNetDetector(BaseDeepfakeDetector):
    """EfficientNet deepfake detector implementation.

    A deepfake detection model using EfficientNet architecture based on the fornet
    library implementation.

    Attributes:
        device: PyTorch device (CPU/CUDA) for model execution
        model: The EfficientNet model for deepfake detection
        confidence_threshold: Minimum confidence for valid detections
        face_size: Input face size for preprocessing
        transform: Image preprocessing transforms
        face_policy: Face extraction policy ('scale' or 'tight')
        mean: ImageNet normalization mean values
        std: ImageNet normalization std values
    """

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.5,
        device: str = None,
        net_model: str = "EfficientNetAutoAttB4",
        train_db: str = "DFDC",
        face_policy: str = "scale",
        face_size: int = 224,
    ):
        """Initializes the EfficientNet deepfake detector.

        Args:
            model_path: Path to the trained model checkpoint (if None, downloads from fornet)
            confidence_threshold: Minimum confidence threshold for detections
            device: Device to run inference on ('cpu' or 'cuda'). Auto-detected if None
            net_model: EfficientNet model variant ('EfficientNetB4', 'EfficientNetAutoAttB4', etc.)
            train_db: Training database ('DFDC', 'FFPP')
            face_policy: Face extraction policy ('scale' or 'tight')
            face_size: Input face size for preprocessing
        """
        super().__init__(confidence_threshold)

        # Set device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model configuration
        self.net_model = net_model
        self.train_db = train_db
        self.face_policy = face_policy
        self.face_size = face_size

        # Initialize model
        self.model = self._create_model()

        # Initialize transforms
        self.transform = self._get_transformer()

    def _create_model(self) -> nn.Module:
        """Creates the EfficientNet model architecture.

        Returns:
            EfficientNet model configured for binary classification
        """
        model_key = "{:s}_{:s}".format(self.net_model, self.train_db)
        model_url = weights.weight_url[model_key]
        net = getattr(fornet, self.net_model)().eval().to(self.device)

        try:
            # Try to load from original URL first
            net.load_state_dict(
                load_url(model_url, map_location=self.device, check_hash=True)
            )

        except Exception as e:
            print(f"Failed to download from original URL ({model_url}): {e}")
            print("Attempting to download from Hugging Face Hub as fallback...")

            try:
                from mukh.core.model_hub import download_efficientnet_model

                # Try to download from Hugging Face Hub
                model_path = download_efficientnet_model(model_key)

                # Load the downloaded model
                state_dict = torch.load(model_path, map_location=self.device)
                net.load_state_dict(state_dict)
                print(
                    f"✅ Successfully loaded model from Hugging Face Hub: {model_path}"
                )

            except Exception as hf_error:
                raise Exception(
                    f"Failed to load model from both original URL and Hugging Face Hub.\n"
                    f"Original error: {e}\n"
                    f"Hugging Face error: {hf_error}"
                )

        return net

    def _get_transformer(self) -> transforms.Compose:
        """Gets the image transformer based on face policy and model normalizer.

        Returns:
            Composed transforms for image preprocessing
        """
        from mukh.deepfake_detection.models.efficientnet.isplutils import utils

        return utils.get_transformer(
            self.face_policy, self.face_size, self.model.get_normalizer(), train=False
        )

    def _extract_face_blazeface(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face using BlazeFace detector (if available).

        Args:
            image: Input image as numpy array.

        Returns:
            numpy.ndarray: Cropped face image, or None if no face found.
        """
        try:
            from mukh.core import download_blazeface_models
            from mukh.face_detection.models.blazeface import BlazeFace, FaceExtractor

            # Initialize BlazeFace if not already done
            if not hasattr(self, "face_extractor"):
                facedet = BlazeFace().to(self.device)
                # Download and get paths for the BlazeFace models
                weights_path, anchors_path = download_blazeface_models()
                facedet.load_weights(weights_path)
                facedet.load_anchors(anchors_path)
                self.face_extractor = FaceExtractor(facedet=facedet)

            # Convert numpy to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Extract faces
            faces_data = self.face_extractor.process_image(img=image)

            if faces_data["faces"]:
                # Return the face with highest confidence
                return faces_data["faces"][0]
            else:
                return None

        except ImportError:
            print("BlazeFace not available, skipping BlazeFace extraction")
            return None
        except Exception as e:
            print(f"BlazeFace extraction failed: {e}")
            return None

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocesses image for model input.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed tensor ready for model input
        """
        # Extract face using BlazeFace
        face_image = self._extract_face_blazeface(image)

        # Apply transforms
        tensor = self.transform(image=np.array(face_image))["image"]

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

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
        """
        # Load image
        image = self._load_image(image_path)

        # Preprocess image
        input_tensor = self._preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)

            # Handle different output formats
            if isinstance(output, tuple):
                logits = output[1] if len(output) > 1 else output[0]
            else:
                logits = output

            prob = torch.sigmoid(logits).item()
            is_deepfake = prob >= self.confidence_threshold
            confidence = prob if is_deepfake else (1 - prob)
            confidence = round(confidence, 2)

            detection = DeepfakeDetection(
                frame_number=0,  # Single image, so frame 0
                is_deepfake=is_deepfake,
                confidence=confidence,
                model_name=f"{self.net_model}",
            )

        # Save to CSV
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
                # Preprocess frame
                input_tensor = self._preprocess_image(frame)

                # Make prediction
                with torch.no_grad():
                    output = self.model(input_tensor)

                    # Handle different output formats
                    if isinstance(output, tuple):
                        logits = output[1] if len(output) > 1 else output[0]
                    else:
                        logits = output

                    prob = torch.sigmoid(logits).item()
                    is_deepfake = prob >= self.confidence_threshold
                    confidence = prob if is_deepfake else (1 - prob)
                    confidence = round(confidence, 2)

                    detection = DeepfakeDetection(
                        frame_number=frame_number,
                        is_deepfake=is_deepfake,
                        confidence=confidence,
                        model_name=f"{self.net_model}",
                    )

                    detections.append(detection)

            except Exception as e:
                print(f"Error processing frame {frame_number}: {e}")
                # Skip frames with errors

        # Aggregate results and print final decision
        if detections:
            final_result, deepfake_count, total_frames = (
                self.aggregate_video_detections(
                    detections, video_path, output_folder, self.net_model
                )
            )

        # Save annotated video
        if save_annotated and detections:
            self._save_annotated_video(video_path, detections, output_folder)

        # Save to CSV
        if save_csv and detections:
            self._save_detections_to_csv(detections, video_path, csv_path)
            self._save_final_video_result_to_txt(
                final_result,
                video_path,
                output_folder,
                self.net_model,
                deepfake_count,
                total_frames,
            )

        return detections, final_result
