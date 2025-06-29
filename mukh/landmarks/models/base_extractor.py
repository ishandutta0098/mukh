"""Base class defining the interface for landmark extraction implementations.

This module provides the abstract base class that all landmark extractor implementations
must inherit from, ensuring a consistent interface across different models.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import cv2
import numpy as np

from ...core.types import FaceDetection


class BaseLandmarkExtractor(ABC):
    """Abstract base class for landmark extractor implementations.

    All landmark extractor implementations must inherit from this class and implement
    the required abstract methods.

    Attributes:
        confidence_threshold: Float threshold (0-1) for detection confidence.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """Initializes the landmark extractor.

        Args:
            confidence_threshold: Minimum confidence threshold for face detections.
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

    def _load_video(self, video_path: str) -> cv2.VideoCapture:
        """Loads a video from disk.

        Args:
            video_path: Path to the video file.

        Returns:
            cv2.VideoCapture: The loaded video capture object.

        Raises:
            ValueError: If the video cannot be loaded from the given path.
        """
        if not os.path.exists(video_path):
            raise ValueError(f"Video path does not exist: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not load video from: {video_path}")

        return cap

    @abstractmethod
    def extract_from_image(
        self,
        image_path: str,
        save_json: bool = True,
        json_path: str = "landmarks.json",
        save_annotated: bool = False,
        output_folder: str = "output",
    ) -> List[np.ndarray]:
        """Extracts landmarks from faces in the given image.

        Args:
            image_path: Path to the input image.
            save_json: Whether to save landmark results to JSON file, defaults to True.
            json_path: Path where to save the JSON file.
            save_annotated: Whether to save annotated image with landmarks.
            output_folder: Folder path where to save annotated images.

        Returns:
            List of numpy arrays containing landmark coordinates for each detected face.
            Each array has shape (num_landmarks, 2) with (x, y) coordinates.
        """
        pass

    @abstractmethod
    def extract_from_array(self, image: np.ndarray) -> List[np.ndarray]:
        """Extracts landmarks from faces in the given image array.

        Args:
            image: Input image as numpy array in BGR format.

        Returns:
            List of numpy arrays containing landmark coordinates for each detected face.
            Each array has shape (num_landmarks, 2) with (x, y) coordinates.
        """
        pass

    def extract_from_video(
        self,
        video_path: str,
        save_json: bool = True,
        json_path: str = "video_landmarks.json",
        save_annotated: bool = False,
        output_folder: str = "output",
        frame_interval: int = 1,
    ) -> List[List[np.ndarray]]:
        """Extracts landmarks from faces in the given video.

        Args:
            video_path: Path to the input video.
            save_json: Whether to save landmark results to JSON file.
            json_path: Path where to save the JSON file.
            save_annotated: Whether to save annotated video with landmarks.
            output_folder: Folder path where to save annotated videos.
            frame_interval: Interval between frames to analyze (default: 1, every frame).

        Returns:
            List of lists containing landmark coordinates for each frame.
            Each frame contains a list of landmark arrays for detected faces.

        Raises:
            ValueError: If the video cannot be loaded.
        """
        cap = self._load_video(video_path)

        all_frame_landmarks = []
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame at specified intervals
                if frame_count % frame_interval == 0:
                    landmarks = self.extract_from_array(frame)
                    all_frame_landmarks.append(landmarks)

                frame_count += 1

        finally:
            cap.release()

        # Save results if requested
        if save_json:
            self._save_video_landmarks_to_json(
                all_frame_landmarks, video_path, json_path
            )

        if save_annotated:
            self._save_annotated_video(
                video_path, all_frame_landmarks, output_folder, frame_interval
            )

        return all_frame_landmarks

    def extract(
        self,
        media_path: str,
        save_json: bool = True,
        json_path: Optional[str] = None,
        save_annotated: bool = False,
        output_folder: str = "output",
        frame_interval: int = 1,
    ) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
        """Extracts landmarks from the given image or video.

        Args:
            media_path: Path to the input image or video.
            save_json: Whether to save landmark results to JSON file.
            json_path: Path where to save the JSON file. If None, auto-generated.
            save_annotated: Whether to save annotated media with landmarks.
            output_folder: Folder path where to save annotated media.
            frame_interval: Interval between frames to analyze for videos.

        Returns:
            For images: List of numpy arrays with landmark coordinates for each face.
            For videos: List of lists with landmark coordinates for each frame and face.

        Raises:
            ValueError: If the media file format is not supported.
        """
        # Determine if input is image or video based on file extension
        _, ext = os.path.splitext(media_path.lower())

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

        # Auto-generate JSON path if not provided
        if json_path is None:
            base_name = os.path.splitext(os.path.basename(media_path))[0]
            if ext in image_extensions:
                json_path = f"{base_name}_landmarks.json"
            else:
                json_path = f"{base_name}_video_landmarks.json"

        if ext in image_extensions:
            return self.extract_from_image(
                media_path, save_json, json_path, save_annotated, output_folder
            )
        elif ext in video_extensions:
            return self.extract_from_video(
                media_path,
                save_json,
                json_path,
                save_annotated,
                output_folder,
                frame_interval,
            )
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _save_landmarks_to_json(
        self,
        landmarks_list: List[np.ndarray],
        image_path: str,
        json_path: str,
    ) -> None:
        """Saves landmark extraction results to a JSON file.

        Args:
            landmarks_list: List of landmark arrays for each detected face.
            image_path: Path to the source image.
            json_path: Path where to save the JSON file.
        """
        os.makedirs(
            os.path.dirname(json_path) if os.path.dirname(json_path) else ".",
            exist_ok=True,
        )

        results = {
            "image_path": image_path,
            "num_faces": len(landmarks_list),
            "landmarks": [],
        }

        for i, landmarks in enumerate(landmarks_list):
            face_landmarks = {
                "face_id": i,
                "num_landmarks": len(landmarks),
                "points": landmarks.tolist(),  # Convert numpy array to list for JSON
            }
            results["landmarks"].append(face_landmarks)

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

    def _save_video_landmarks_to_json(
        self,
        all_frame_landmarks: List[List[np.ndarray]],
        video_path: str,
        json_path: str,
    ) -> None:
        """Saves video landmark extraction results to a JSON file.

        Args:
            all_frame_landmarks: List of landmark lists for each frame.
            video_path: Path to the source video.
            json_path: Path where to save the JSON file.
        """
        os.makedirs(
            os.path.dirname(json_path) if os.path.dirname(json_path) else ".",
            exist_ok=True,
        )

        results = {
            "video_path": video_path,
            "num_frames": len(all_frame_landmarks),
            "frames": [],
        }

        for frame_idx, frame_landmarks in enumerate(all_frame_landmarks):
            frame_data = {
                "frame_number": frame_idx,
                "num_faces": len(frame_landmarks),
                "landmarks": [],
            }

            for face_idx, landmarks in enumerate(frame_landmarks):
                face_landmarks = {
                    "face_id": face_idx,
                    "num_landmarks": len(landmarks),
                    "points": landmarks.tolist(),
                }
                frame_data["landmarks"].append(face_landmarks)

            results["frames"].append(frame_data)

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

    def _draw_landmarks(
        self, image: np.ndarray, landmarks_list: List[np.ndarray]
    ) -> np.ndarray:
        """Draws landmarks on the image.

        Args:
            image: Input image as numpy array.
            landmarks_list: List of landmark arrays for each face.

        Returns:
            np.ndarray: Copy of input image with landmarks drawn.
        """
        annotated_image = image.copy()

        for landmarks in landmarks_list:
            # Draw each landmark point
            for x, y in landmarks:
                cv2.circle(
                    annotated_image,
                    (int(x), int(y)),
                    radius=2,
                    color=(0, 255, 0),  # Green color
                    thickness=-1,
                )

        return annotated_image

    def _save_annotated_image(
        self,
        image_path: str,
        landmarks_list: List[np.ndarray],
        output_folder: str,
    ) -> str:
        """Saves annotated image with landmark results.

        Args:
            image_path: Path to the original image.
            landmarks_list: List of landmark arrays for each face.
            output_folder: Folder where to save the annotated image.

        Returns:
            str: Path to the saved annotated image.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Load the original image
        image = self._load_image(image_path)

        # Draw landmarks on image
        annotated_image = self._draw_landmarks(image, landmarks_list)

        # Create output filename
        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        output_filename = f"{name}_landmarks{ext}"
        output_path = os.path.join(output_folder, output_filename)

        # Save annotated image
        cv2.imwrite(output_path, annotated_image)

        return output_path

    def _save_annotated_video(
        self,
        video_path: str,
        all_frame_landmarks: List[List[np.ndarray]],
        output_folder: str,
        frame_interval: int = 1,
    ) -> str:
        """Saves annotated video with landmark results.

        Args:
            video_path: Path to the original video.
            all_frame_landmarks: List of landmark lists for each processed frame.
            output_folder: Folder where to save the annotated video.
            frame_interval: Interval between frames that were processed.

        Returns:
            str: Path to the saved annotated video.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Open input video
        cap = self._load_video(video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output filename
        video_name = os.path.basename(video_path)
        name, ext = os.path.splitext(video_name)
        output_filename = f"{name}_landmarks{ext}"
        output_path = os.path.join(output_folder, output_filename)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            frame_count = 0
            landmarks_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw landmarks on frames that were processed
                if frame_count % frame_interval == 0 and landmarks_idx < len(
                    all_frame_landmarks
                ):
                    frame = self._draw_landmarks(
                        frame, all_frame_landmarks[landmarks_idx]
                    )
                    landmarks_idx += 1

                out.write(frame)
                frame_count += 1

        finally:
            cap.release()
            out.release()

        return output_path
