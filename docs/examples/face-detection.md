# Face Detection Examples

## Basic Detection

```python
from mukh.face_detection import FaceDetector

detector = FaceDetector.create("mediapipe")  # Options: "blazeface", "mediapipe", "ultralight"

detections = detector.detect(
    image_path="assets/images/img1.jpg",
    save_csv=True,
    csv_path="output/mediapipe/detections.csv",
    save_annotated=True,
    output_folder="output/mediapipe"
)
```

## Available Models

- `blazeface`
- `mediapipe` 
- `ultralight` 