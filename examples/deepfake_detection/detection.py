"""
Basic example showing how to use deepfake detector to detect deepfakes in media files.

Usage:
For images:
python examples/deepfake_detection/detection.py \
  --detection_model resnet_inception \
  --media_path assets/images/img1.jpg \
  --confidence_threshold 0.5

For videos:
python examples/deepfake_detection/detection.py \
  --detection_model resnet_inception \
  --media_path assets/videos/deepfake_elon_musk.mp4 \
  --confidence_threshold 0.5 \
  --num_frames 11
"""

import argparse

import torch

from mukh.deepfake_detection import DeepfakeDetector

parser = argparse.ArgumentParser(description="Deepfake Detection Example")
parser.add_argument(
    "--detection_model",
    type=str,
    choices=["resnet_inception", "efficientnet"],
    default="resnet_inception",
    help="Choose the deepfake detection model to use.",
)
parser.add_argument(
    "--media_path",
    type=str,
    default="assets/images/img1.jpg",
    help="Path to the media file (image or video) to analyze for deepfakes.",
)
parser.add_argument(
    "--confidence_threshold",
    type=float,
    default=0.5,
    help="Confidence threshold for deepfake detection (0.0 to 1.0).",
)
parser.add_argument(
    "--num_frames",
    type=int,
    default=11,
    help="Number of equally spaced frames to extract from video for analysis.",
)

args = parser.parse_args()

# Create detector
detector = DeepfakeDetector(
    model_name=args.detection_model,
    confidence_threshold=args.confidence_threshold,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Detect deepfakes in the media file
detections, final_result = detector.detect(
    media_path=args.media_path,  # Path to the media file to analyze (image/video)
    save_csv=True,  # Save the detections to a CSV file
    csv_path=f"output/{args.detection_model}/deepfake_detections.csv",  # Path to save the CSV file
    save_annotated=True,  # Save the annotated media
    output_folder=f"output/{args.detection_model}",  # Path to save the annotated media
    num_frames=args.num_frames,  # Number of equally spaced frames for video analysis
)

print(f"Deepfake: {final_result}")
