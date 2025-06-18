"""
Basic example showing how to use a single face detector to detect faces in an image.

Usage:
python examples/face_detection/basic_detection.py \
    --detection_model mediapipe \
    --image_path assets/images/img1.jpg
"""


import argparse

from mukh.face_detection import FaceDetector

parser = argparse.ArgumentParser(description="Face Detection Example")
parser.add_argument(
    "--detection_model",
    type=str,
    choices=["blazeface", "mediapipe", "ultralight"],
    default="ultralight",
    help="Choose the face detection model to use.",
)
parser.add_argument(
    "--image_path",
    type=str,
    default="assets/images/img1.jpg",
    help="Path to the image to detect faces in.",
)

args = parser.parse_args()

# Create detector
detector = FaceDetector.create(args.detection_model)

detections = detector.detect(
    image_path=args.image_path,  # Path to the image to detect faces in
    save_csv=True,  # Save the detections to a CSV file
    csv_path=f"output/{args.detection_model}/detections.csv",  # Path to save the CSV file
    save_annotated=True,  # Save the annotated image
    output_folder=f"output/{args.detection_model}",  # Path to save the annotated image
)
