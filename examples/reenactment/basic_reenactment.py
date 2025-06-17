"""
Basic example showing how to use a single face reenactor to reenact a face from an image.

Usage:
python examples/reenactment/basic_reenactment.py \
  --reenactor_model tps \
  --source_path assets/images/img1.jpg \
  --driving_video_path assets/videos/video_1sec.mp4 \
  --output_folder output
"""

import argparse

from mukh.reenactment import FaceReenactor

parser = argparse.ArgumentParser(description="Face Reenactment Example")
parser.add_argument(
    "--reenactor_model",
    type=str,
    choices=["tps"],
    default="tps",
    help="Choose the reenactor model to use.",
)
parser.add_argument(
    "--source_path",
    type=str,
    default="assets/images/img1.jpg",
    help="Path to the source image.",
)
parser.add_argument(
    "--driving_video_path",
    type=str,
    default="assets/videos/video_1sec.mp4",
    help="Path to the driving video.",
)
parser.add_argument(
    "--output_folder",
    type=str,
    default="output",
    help="Path to save the reenacted video.",
)

args = parser.parse_args()

reenactor = FaceReenactor.create(args.reenactor_model)

result_path = reenactor.reenact_from_video(
    source_path=args.source_path,
    driving_video_path=args.driving_video_path,
    output_path=f"{args.output_folder}/{args.reenactor_model}",
    save_comparison=True,
    resize_to_image_resolution=False,
)

print(f"Reenacted video saved to: {result_path}")
