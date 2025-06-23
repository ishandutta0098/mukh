"""
This script demonstrates how to use the FaceDetector class to detect faces in a folder of images.
It uses the  multiprocessing to detect faces in all images in the folder.
It also saves the detections to a JSON file and annotated images to a folder.
By default it runs sequentially by using num_processes=0.
If you want to run it in parallel, you can set num_processes to the number of processes you want to use.

Use mukh.utils.parallel.get_cpu_count() to get the number of CPU cores available.

Recommended num_processes for models other than MediaPipe is CPU_COUNT() - 1.

NOTE: 
In case of MediaPipe, recommended num_processes is 0.
MediaPipe face detection does not work reliably with multiprocessing due to OpenGL context issues.

Example usage:
python basic_detection_folder.py \
    --detection_model blazeface \
    --images_folder assets/images \
    --output_folder output \
    --json_path output/detections.json \
    --num_processes 0
"""

import argparse

from mukh.face_detection import FaceDetector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face Detection Example - Batch Processing"
    )
    parser.add_argument(
        "--detection_model",
        type=str,
        choices=["blazeface", "mediapipe", "ultralight"],
        default="blazeface",
        help="Choose the face detection model to use.",
    )
    parser.add_argument(
        "--images_folder",
        type=str,
        default="assets/images",
        help="Path to the folder containing the images to detect faces in.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Path to save the output files.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="output/detections.json",
        help="Path to save the consolidated JSON file with all detections.",
    )
    parser.add_argument(
        "--save_annotated",
        action="store_true",
        help="Save annotated images with bounding boxes.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=0,
        help="Number of processes to use for parallel processing. Defaults to 0. Will run sequentially if set to 0.",
    )

    args = parser.parse_args()

    # Create the face detector
    detector = FaceDetector.create(args.detection_model)

    # Process all images in the folder
    print(
        f"Processing images from {args.images_folder} using {args.detection_model} model..."
    )

    detections = detector.detect_folder(
        images_folder=args.images_folder,
        output_folder=args.output_folder,
        save_json=True,
        json_path=args.json_path,
        save_annotated=args.save_annotated,
        num_processes=args.num_processes,
        detector_model=args.detection_model,
    )

    print(f"Completed! Processed {len(detections)} total detections across all images.")
    print(f"Results saved to: {args.json_path}")
    if args.save_annotated:
        print(f"Annotated images saved to: {args.output_folder}")
