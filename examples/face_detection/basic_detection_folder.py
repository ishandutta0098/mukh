import os
import argparse
from functools import partial
from mukh.face_detection import FaceDetector
from mukh.utils.parallel import tqdm_parallel_processor

def process_image(image: str, detection_model: str, images_folder: str, output_folder: str) -> None:
    """
    Process a single image for face detection.
    Args:
        image: The filename of the image to process.
        detection_model: The face detection model to use.
        images_folder: Path to the folder containing the input images.
        output_folder: Path to save the output.
    """
    # Initialize the detector inside the worker process
    local_detector = FaceDetector.create(detection_model)

    # Construct the full path to the image
    image_path = os.path.join(images_folder, image)
    output_csv_path = os.path.join(output_folder, image.split('.')[0], "detections.csv")
    output_image_folder = os.path.join(output_folder, image.split('.')[0])

    # Perform face detection
    local_detector.detect(
        image_path=image_path,
        save_csv=True,
        csv_path=output_csv_path,
        save_annotated=True,
        output_folder=output_image_folder,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Detection Example")
    parser.add_argument(
        "--detection_model",
        type=str,
        choices=["blazeface", "mediapipe", "ultralight"],
        default="ultralight",
        help="Choose the face detection model to use.",
    )
    parser.add_argument(
        "--images_folder",
        type=str,
        default="data/demo_fake_extracted/fake_car_show_all_frames",
        help="Path to the folder containing the images to detect faces in.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output/mediapipe/fake_car_show_all_frames",
        help="Path to save the annotated image.",
    )

    args = parser.parse_args()

    # Get all images in the folder
    images = [f for f in os.listdir(args.images_folder) if f.endswith((".jpg", ".png"))]

    # Create a partial function with the fixed arguments
    process_image_partial = partial(
        process_image,
        detection_model=args.detection_model,
        images_folder=args.images_folder,
        output_folder=args.output_folder,
    )

    # Process images in parallel
    tqdm_parallel_processor(
        function=process_image_partial,
        iterable=images,
        description="Processing images",
        num_processes=os.cpu_count(),
    )

# import argparse
# import os

# from mukh.face_detection import FaceDetector
# from tqdm import tqdm

# parser = argparse.ArgumentParser(description="Face Detection Example")
# parser.add_argument(
#     "--detection_model",
#     type=str,
#     choices=["blazeface", "mediapipe", "ultralight"],
#     default="mediapipe",
#     help="Choose the face detection model to use.",
# )

# args = parser.parse_args()

# # Create detector
# detector = FaceDetector.create(args.detection_model)

# # Process all images in the demo_images folder
# images_folder = "data/demo_fake_extracted/fake_car_show_all_frames"
# for image_name in tqdm(os.listdir(images_folder)):
#     if image_name.endswith((".jpg", ".png")):
#         # Get image path
#         image_path = os.path.join(images_folder, image_name)

#         # Detect faces
#         detections = detector.detect(
#             image_path=image_path,  # Path to the image to detect faces in
#             save_csv=True,  # Save the detections to a CSV file
#             csv_path=f"output/{args.detection_model}/{image_name.split('.')[0]}/detections.csv",  # Path to save the CSV file
#             save_annotated=True,  # Save the annotated image
#             output_folder=f"output/{args.detection_model}/{image_name.split('.')[0]}",  # Path to save the annotated image
#         )