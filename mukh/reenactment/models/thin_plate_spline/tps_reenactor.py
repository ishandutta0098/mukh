"""Implementation of face reenactment using Thin Plate Spline model.

Source: https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model

LICENSE: MIT
"""

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
from skimage import img_as_ubyte
from skimage.transform import resize

from mukh.reenactment.models.base_reenactor import BaseFaceReenactor
from mukh.reenactment.models.thin_plate_spline.utils import (
    find_best_frame,
    load_checkpoints,
    make_animation,
)


class ThinPlateSplineReenactor(BaseFaceReenactor):
    """Implements face reenactment using the Thin Plate Spline model.

    This class uses the TPS model to perform face reenactment, transferring facial
    expressions from a driving image/video to a source image.

    Attributes:
        model_path: Path to the model checkpoint file.
        device: Device to run inference on ('cpu', 'cuda', etc.).
        config_path: Path to the model configuration file.
        predict_mode: Animation prediction mode ('standard', 'relative', 'avd').
        find_best_frame: Whether to find the best frame when using relative mode.
        pixel: Resolution to resize images to (default: 256).
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "cpu",
        predict_mode: str = "relative",
        find_best_frame: bool = False,
        pixel: int = 256,
    ):
        """Initializes the TPS face reenactor.

        Args:
            model_path: Path to the model checkpoint file.
            config_path: Path to the model configuration file.
            device: Device to run inference on ('cpu', 'cuda'). Defaults to 'cpu'.
            predict_mode: Animation prediction mode ('standard', 'relative', 'avd').
                Defaults to 'relative'.
            find_best_frame: Whether to find the best frame when using relative mode.
                Defaults to True.
            pixel: Resolution to resize images to. Defaults to 256.
        """
        super().__init__(model_path, device)
        self.config_path = config_path
        self.predict_mode = predict_mode
        self.find_best_frame = find_best_frame
        self.pixel = pixel
        self.device = torch.device(device)

        # Initialize model components to None
        self.inpainting = None
        self.kp_detector = None
        self.dense_motion_network = None
        self.avd_network = None

        # Load the model during initialization
        self._load_model()

    def _load_model(self) -> None:
        """Loads the TPS reenactment model components.

        Loads inpainting, keypoint detector, dense motion network, and AVD network
        from the specified checkpoint path.

        Raises:
            ValueError: If the model cannot be loaded from the given path.
        """
        try:
            (
                self.inpainting,
                self.kp_detector,
                self.dense_motion_network,
                self.avd_network,
            ) = load_checkpoints(
                config_path=self.config_path,
                checkpoint_path=self.model_path,
                device=self.device,
            )
        except Exception as e:
            raise ValueError(f"Failed to load model from {self.model_path}: {str(e)}")

    def _read_image(self, image_path: str) -> np.ndarray:
        """Reads and preprocesses an image from a file path.

        Args:
            image_path: Path to the image file.

        Returns:
            Preprocessed image as a numpy array.

        Raises:
            ValueError: If the image cannot be read.
        """
        try:
            image = imageio.imread(image_path)
            return resize(image, (self.pixel, self.pixel))[..., :3]
        except Exception as e:
            raise ValueError(f"Failed to read image from {image_path}: {str(e)}")

    def _read_video(self, video_path: str) -> Tuple[list, float]:
        """Reads and preprocesses a video from a file path.

        Args:
            video_path: Path to the video file.

        Returns:
            Tuple containing list of preprocessed frames and the video FPS.

        Raises:
            ValueError: If the video cannot be read.
        """
        try:
            reader = imageio.get_reader(video_path)
            fps = reader.get_meta_data()["fps"]
            frames = []

            try:
                for im in reader:
                    frames.append(im)
            except RuntimeError:
                pass
            finally:
                reader.close()

            frames = [
                resize(frame, (self.pixel, self.pixel))[..., :3] for frame in frames
            ]
            return frames, fps
        except Exception as e:
            raise ValueError(f"Failed to read video from {video_path}: {str(e)}")

    def _postprocess(self, predictions: list, original_shape: Tuple[int, int]) -> list:
        """Postprocesses the generated frames to match original image resolution.

        Args:
            predictions: List of generated frames from the model.
            original_shape: Original shape (height, width) to resize to.

        Returns:
            List of postprocessed frames.
        """
        if predictions[0].shape[:2] == original_shape:
            return predictions

        postprocessed = []
        for frame in predictions:
            # Resize to original resolution
            resized_frame = resize(frame, original_shape, anti_aliasing=True)
            postprocessed.append(resized_frame)

        return postprocessed

    def _save_comparison_animation(
        self,
        source_image: np.ndarray,
        driving_video: list,
        predictions: list,
        output_path: str,
        fps: float,
    ) -> str:
        """Creates and saves a comparison animation of source, driving, and generated frames.

        Args:
            source_image: Source image used for reenactment
            driving_video: List of driving video frames
            predictions: List of generated frames
            output_path: Path where to save the comparison animation
            fps: Frames per second for the animation

        Returns:
            str: Path to the saved comparison animation
        """
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        # Resize all inputs to match dimensions
        target_shape = predictions[0].shape[:2]
        resized_source = resize(source_image, target_shape, anti_aliasing=True)
        resized_driving = [
            resize(frame, target_shape, anti_aliasing=True) for frame in driving_video
        ]

        fig = plt.figure(figsize=(12, 6))

        ims = []
        for i in range(len(resized_driving)):
            cols = [resized_source]
            cols.append(resized_driving[i])
            cols.append(predictions[i])
            im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
            plt.axis("off")
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)

        # Save the comparison animation
        ani.save(output_path, writer="ffmpeg", fps=fps)
        plt.close()

        return output_path

    def reenact_from_video(
        self,
        source_path: str,
        driving_video_path: str,
        output_path: Optional[str] = "output",
        save_comparison: bool = False,
        resize_to_image_resolution: bool = True,
    ) -> str:
        """Performs face reenactment using a source image and driving video.

        Args:
            source_path: Path to the source image (face to be animated).
            driving_video_path: Path to the driving video (facial motion to transfer).
            output_path: Optional path to the output directory. Defaults to "output".
            save_comparison: Whether to save a comparison animation showing source,
                driving, and generated frames side by side. Defaults to False.
            resize_to_image_resolution: Whether to resize the output video to match
                the original source image resolution. Defaults to True.

        Returns:
            str: Path to the generated output video.
        """
        # Read the original source image to get its dimensions
        try:
            original_source = imageio.imread(source_path)
            original_shape = original_source.shape[:2]  # (height, width)
        except Exception as e:
            raise ValueError(
                f"Failed to read original source image from {source_path}: {str(e)}"
            )

        source_image = self._read_image(source_path)
        driving_video, fps = self._read_video(driving_video_path)

        # Extract filenames without extensions for both source and driving
        source_name = os.path.splitext(os.path.basename(source_path))[0]
        driving_name = os.path.splitext(os.path.basename(driving_video_path))[0]

        # Make sure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Create full output file paths
        output_video_path = os.path.join(
            output_path, f"reenacted_{source_name}_by_{driving_name}.mp4"
        )

        # Perform reenactment based on predict_mode and find_best_frame settings
        if self.predict_mode == "relative" and self.find_best_frame:
            i = find_best_frame(source_image, driving_video, self.device.type == "cpu")

            driving_forward = driving_video[i:]
            driving_backward = driving_video[: (i + 1)][::-1]

            predictions_forward = make_animation(
                source_image,
                driving_forward,
                self.inpainting,
                self.kp_detector,
                self.dense_motion_network,
                self.avd_network,
                device=self.device,
                mode=self.predict_mode,
            )

            predictions_backward = make_animation(
                source_image,
                driving_backward,
                self.inpainting,
                self.kp_detector,
                self.dense_motion_network,
                self.avd_network,
                device=self.device,
                mode=self.predict_mode,
            )

            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = make_animation(
                source_image,
                driving_video,
                self.inpainting,
                self.kp_detector,
                self.dense_motion_network,
                self.avd_network,
                device=self.device,
                mode=self.predict_mode,
            )

        # Postprocess predictions if needed
        if resize_to_image_resolution:
            predictions = self._postprocess(predictions, original_shape)

        # Save the resulting video
        imageio.mimsave(
            output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps
        )

        # Optionally save comparison animation
        if save_comparison:
            # Create comparison output path
            comparison_path = os.path.join(
                output_path, f"comparison_{source_name}_by_{driving_name}.mp4"
            )

            # Create and save the comparison animation
            self._save_comparison_animation(
                source_image, driving_video, predictions, comparison_path, fps
            )

        return output_video_path
