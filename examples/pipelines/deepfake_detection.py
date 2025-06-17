"""
Example script demonstrating how to use the PipelineDeepfakeDetection class.

This script shows various ways to use the ensemble deepfake detection pipeline
with different model configurations and settings.

Usage:
python examples/pipelines/deepfake_detection.py \
  --media_path assets/videos/deepfake_elon_musk.mp4 \
  --output_folder output/deepfake_detection_pipeline
"""

import argparse
import os
import sys
from pathlib import Path

from mukh.pipelines.deepfake_detection import PipelineDeepfakeDetection

parser = argparse.ArgumentParser(description="Deepfake Detection Pipeline")
parser.add_argument(
    "--media_path",
    type=str,
    help="Path to the media file to analyze",
    default="assets/videos/deepfake_elon_musk.mp4",
)
parser.add_argument(
    "--output_folder",
    type=str,
    help="Path to the output folder",
    default="output/deepfake_detection_pipeline",
)
args = parser.parse_args()


# Define model configurations with weights
model_configs = {"resnet_inception": 0.5, "efficientnet": 0.5}

# Create ensemble detector
detector = PipelineDeepfakeDetection(model_configs)

# Print detector information
print("Detector Info:")
info = detector.get_model_info()
for key, value in info.items():
    print(f"  {key}: {value}")

if os.path.exists(args.media_path):
    try:
        # Run detection
        result = detector.detect(
            media_path=args.media_path,
            output_folder=args.output_folder,
            save_csv=True,
            num_frames=11,
        )

        print(result)
        print(f"\nFinal Result: {'DEEPFAKE' if result else 'REAL'}")
        print(f"Results saved to: {args.output_folder}")

    except Exception as e:
        print(f"Error during detection: {e}")
else:
    print(f"Media file not found: {args.media_path}")
    print("Please update the media_path variable with a valid file path.")
