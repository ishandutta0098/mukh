"""Base class defining the interface for face detection implementations.

This module provides the abstract base class that all face detector implementations
must inherit from, ensuring a consistent interface across different models.
"""

import json
import os
import platform
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from ...core.types import FaceDetection


def _process_single_image_for_batch_worker(args_tuple: tuple) -> List[dict]:
    """Worker function for batch processing - must be at module level for pickling.

    Args:
        args_tuple: Tuple containing (image_filename, images_folder, output_folder,
                   save_annotated, detector_class_name, detector_kwargs)

    Returns:
        List of detection dictionaries for this image.
    """
    (
        image_filename,
        images_folder,
        output_folder,
        save_annotated,
        detector_class_name,
        detector_kwargs,
    ) = args_tuple

    try:
        # Import the detector factory here to avoid circular imports
        from .. import FaceDetector

        # Create a new detector instance for this process
        detector = FaceDetector.create(detector_class_name)

        # Construct paths
        image_path = os.path.join(images_folder, image_filename)

        # Set up output paths for annotated images if requested
        if save_annotated:
            image_output_folder = os.path.join(
                output_folder, os.path.splitext(image_filename)[0]
            )
        else:
            image_output_folder = output_folder

        # Perform detection (without saving individual JSON files)
        detections = detector.detect(
            image_path=image_path,
            save_json=False,  # Don't save individual JSON files
            save_annotated=save_annotated,
            output_folder=image_output_folder,
        )

        # Convert detections to dictionary format for JSON serialization
        detection_results = []
        for detection in detections:
            bbox = detection.bbox
            detection_results.append(
                {
                    "image_name": image_filename,
                    "x1": bbox.x1,
                    "y1": bbox.y1,
                    "x2": bbox.x2,
                    "y2": bbox.y2,
                    "confidence": bbox.confidence,
                }
            )

        return detection_results

    except Exception as e:
        print(f"Error processing {image_filename}: {str(e)}")
        return []


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

    def detect_folder(
        self,
        images_folder: str,
        output_folder: str = "output",
        save_json: bool = True,
        json_path: str = "batch_detections.json",
        save_annotated: bool = False,
        num_processes: Optional[int] = 0,
        image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
        detector_model: str = "mediapipe",
    ) -> List[dict]:
        """Detects faces in a batch of images using multiprocessing.

        This method processes all images in a folder in parallel, collecting all
        detections into a single JSON file. It handles the multiprocessing setup
        and cleanup automatically. If multiprocessing fails, it automatically
        falls back to sequential processing.

        Args:
            images_folder: Path to the folder containing input images.
            output_folder: Path to save the output files.
            save_json: Whether to save detection results to JSON file.
            json_path: Path where to save the consolidated JSON file.
            save_annotated: Whether to save annotated images with bounding boxes.
            num_processes: Number of processes to use. Defaults to 0. Will run sequentially if set to 0.
            image_extensions: Tuple of valid image file extensions.
            detector_model: The detector model type to use for batch processing.

        Returns:
            List of dictionaries containing detection results for all images.

        Raises:
            ValueError: If the images folder doesn't exist or contains no valid images.
        """
        if not os.path.exists(images_folder):
            raise ValueError(f"Images folder does not exist: {images_folder}")

        # Get all valid image files
        images = [
            f for f in os.listdir(images_folder) if f.lower().endswith(image_extensions)
        ]

        if not images:
            raise ValueError(f"No valid images found in {images_folder}")

        print(f"Found {len(images)} images to process")

        # Check for MediaPipe parallel processing
        if detector_model.lower() == "mediapipe" and (
            num_processes is None or num_processes > 1
        ):
            warnings.warn(
                "MediaPipe face detection does not work reliably with multiprocessing due to "
                "OpenGL context issues. Falling back to sequential processing.",
                UserWarning,
                stacklevel=2,
            )
            print("Using sequential processing for MediaPipe model...")
            return self._process_images_sequentially(
                images,
                images_folder,
                output_folder,
                save_annotated,
                save_json,
                json_path,
            )

        # Sequential processing if num_processes is 0 or 1
        if num_processes == 0 or num_processes == 1:
            print("Using sequential processing (num_processes=0)...")
            return self._process_images_sequentially(
                images,
                images_folder,
                output_folder,
                save_annotated,
                save_json,
                json_path,
            )

        # Try multiprocessing
        print(f"Attempting parallel processing with {num_processes} processes...")

        try:
            # Determine the appropriate start method based on the platform
            # On macOS, use "spawn" to avoid OpenGL context issues with fork()
            start_method = "spawn" if platform.system() == "Darwin" else "fork"

            # Import here to avoid circular imports
            from ...utils.parallel import MultiProcessor

            # Prepare arguments for each worker process
            worker_args = [
                (
                    image_filename,
                    images_folder,
                    output_folder,
                    save_annotated,
                    detector_model,
                    {},
                )
                for image_filename in images
            ]

            # Process images in parallel
            processor = MultiProcessor(
                num_processes=num_processes, start_mode=start_method
            )

            results = processor.process(
                function=_process_single_image_for_batch_worker,
                iterable=worker_args,
                description="Processing images for face detection",
            )

            # Consolidate all detections into a single list
            all_detections = []
            for result in results:
                if result:  # Skip None results
                    all_detections.extend(result)

            print(f"Parallel processing completed successfully!")

        except Exception as e:
            print(f"Multiprocessing failed with error: {str(e)}")
            print("Falling back to sequential processing...")
            warnings.warn(
                f"Multiprocessing failed ({str(e)}). Falling back to sequential processing.",
                UserWarning,
                stacklevel=2,
            )

            # Fall back to sequential processing
            return self._process_images_sequentially(
                images,
                images_folder,
                output_folder,
                save_annotated,
                save_json,
                json_path,
            )

        # Save consolidated JSON if requested
        if save_json:
            self._save_folder_detections_to_json(all_detections, json_path)

        return all_detections

    def _process_images_sequentially(
        self,
        images: List[str],
        images_folder: str,
        output_folder: str,
        save_annotated: bool,
        save_json: bool,
        json_path: str,
    ) -> List[dict]:
        """Process images sequentially without multiprocessing.

        Args:
            images: List of image filenames to process.
            images_folder: Path to the folder containing input images.
            output_folder: Path to save the output files.
            save_annotated: Whether to save annotated images.
            save_json: Whether to save detection results to JSON file.
            json_path: Path where to save the consolidated JSON file.

        Returns:
            List of dictionaries containing detection results for all images.
        """
        all_detections = []

        for i, image_filename in tqdm(
            enumerate(images, 1), total=len(images), desc="Processing images"
        ):

            try:
                image_path = os.path.join(images_folder, image_filename)

                # Set up output paths for annotated images if requested
                if save_annotated:
                    image_output_folder = os.path.join(
                        output_folder, os.path.splitext(image_filename)[0]
                    )
                else:
                    image_output_folder = output_folder

                # Perform detection (without saving individual JSON files)
                detections = self.detect(
                    image_path=image_path,
                    save_json=False,  # Don't save individual JSON files
                    save_annotated=save_annotated,
                    output_folder=image_output_folder,
                )

                # Convert detections to dictionary format for JSON serialization
                for detection in detections:
                    bbox = detection.bbox
                    all_detections.append(
                        {
                            "image_name": image_filename,
                            "x1": bbox.x1,
                            "y1": bbox.y1,
                            "x2": bbox.x2,
                            "y2": bbox.y2,
                            "confidence": bbox.confidence,
                        }
                    )

            except Exception as e:
                print(f"Error processing {image_filename}: {str(e)}")

        # Save consolidated JSON if requested
        if save_json:
            self._save_folder_detections_to_json(all_detections, json_path)

        return all_detections

    def _save_folder_detections_to_json(
        self, all_detections: List[dict], json_path: str
    ) -> None:
        """Saves batch detection results to a single JSON file.

        Args:
            all_detections: List of detection dictionaries from all images.
            json_path: Path where to save the consolidated JSON file.
        """
        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(json_path) if os.path.dirname(json_path) else ".",
            exist_ok=True,
        )

        # Write consolidated results to JSON
        with open(json_path, "w", encoding="utf-8") as jsonfile:
            json.dump(all_detections, jsonfile, indent=4)

        print(f"Saved {len(all_detections)} detections to {json_path}")

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
            detection_results.append(
                {
                    "image_name": image_name,
                    "x1": bbox.x1,
                    "y1": bbox.y1,
                    "x2": bbox.x2,
                    "y2": bbox.y2,
                    "confidence": bbox.confidence,
                }
            )

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
