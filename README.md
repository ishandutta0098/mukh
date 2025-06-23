# Mukh

<div align="center">

[![Downloads](https://static.pepy.tech/personalized-badge/mukh?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/mukh)
[![Documentation](https://img.shields.io/badge/docs-View%20Documentation-blue.svg?style=flat)](https://ishandutta0098.github.io/mukh/)
[![Stars](https://img.shields.io/github/stars/ishandutta0098/mukh?color=yellow&style=flat&label=%E2%AD%90%20stars)](https://github.com/ishandutta0098/mukh/stargazers)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg?style=flat)](https://github.com/ishandutta0098/mukh/blob/master/LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-mukh-orange.svg?style=flat&logo=pypi)](https://pypi.org/project/mukh/)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-@ishandutta0098-blue.svg?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ishandutta0098)
[![Twitter](https://img.shields.io/:follow-@ishandutta0098-blue.svg?style=flat&logo=x)](https://twitter.com/intent/user?screen_name=ishandutta0098)
[![YouTube](https://img.shields.io/badge/YouTube-@ishandutta--ai-red?style=flat&logo=youtube)](https://www.youtube.com/@ishandutta-ai)

</div>

Mukh (मुख, meaning "face" in Sanskrit) is a comprehensive face analysis library that provides unified APIs for various face-related tasks. It simplifies the process of working with multiple face analysis models through a consistent interface.

## Features
- 🥸 **DeepFake Detector:** First python package featuring an Ensemble of multiple models
- 🎯 **Unified API**: Single, consistent API for multiple face analysis tasks like face detection and reenactment
- 🔄 **Model Flexibility**: Support for multiple models per task
- 🛠️ **Custom Pipelines**: Optimized preprocessing and model combinations
  

## Documentation
The library is documented in detail, [click here](https://ishandutta0098.github.io/mukh/) to view the documentation.
  
## Currently Supported Tasks

- Face Detection
- Face Reenactment with Source Image and Driving Video
- Deepfake Detection for Image and Video
- Deepfake Detection Pipeline - Ensemble of multiple models

## Installation

```bash
conda create -n mukh-dev python=3.10 -y
conda activate mukh-dev
pip install mukh==0.1.14
```

## Usage

## Face Detection

```python
from mukh.face_detection import FaceDetector

# Initialize detector
detection_model = "mediapipe"                                  # Other models: "blazeface", "ultralight"
detector = FaceDetector.create(detection_model)

# Detect faces
detections = detector.detect(
    image_path="assets/images/img1.jpg",                       # Path to the image to detect faces in
    save_json=True,                                             # Save the detections to a JSON file
    json_path=f"output/{detection_model}/detections.json",       # Path to save the JSON file
    save_annotated=True,                                       # Save the annotated image
    output_folder=f"output/{detection_model}",                   # Path to save the annotated image
)
```
  
### Example
```python 
python examples/face_detection/basic_detection_single_image.py --detection_model mediapipe
```

### Output Annotated Image
<img src = "https://github.com/user-attachments/assets/e2670b72-eb53-4cc1-ba5c-a3293ae23589" width=200> 
  
```python
image_name | x1 |   y1  |  x2 |  y2 | confidence
 img1.jpg  | 62 |  228  | 453 | 619 | 0.9381868243217468
```
  
## Face Reenactment

```python
from mukh.reenactment import FaceReenactor

# Initialize reenactor
reenactor_model = "tps"                           # Available models: "tps"
reenactor = FaceReenactor.create(reenactor_model)

# Reenact face
result_path = reenactor.reenact_from_video(
    source_path="assets/images/img1.jpg",         # Path to the source image
    driving_video_path="assets/videos/video.mp4", # Path to the driving video
    output_path=f"output/{reenactor_model}",      # Path to save the reenacted video
    save_comparison=True,                         # Save the comparison video
    resize_to_image_resolution=False,             # Resize the reenacted video to the image resolution
)
```

### Example
```python
python examples/reenactment/basic_reenactment.py \
  --reenactor_model tps \
  --source_path assets/images/img1.jpg \
  --driving_video_path assets/videos/video_1sec.mp4 \
  --output_folder output
```

### Output
  
https://github.com/user-attachments/assets/875ba692-ea78-42e3-9e03-d1f4703930be
  
## Deepfake Detection

### Images

```python
import torch
from mukh.deepfake_detection import DeepfakeDetector

# Initialize detector
detection_model = "efficientnet"                  # Other models "resnet_inception"

detector = DeepfakeDetector(
    model_name=detection_model,
    confidence_threshold=0.5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Detect deepfakes in Images

media_path = "assets/images/img1.jpg"             

detections, final_result = detector.detect(
    media_path=media_path,                                         # Path to the media file (image/video)
    save_csv=True,                                                 # Save the detections to a CSV file
    csv_path=f"output/{detection_model}/deepfake_detections.csv",  # Path to save the CSV file
    save_annotated=True,                                           # Save the annotated media
    output_folder=f"output/{detection_model}",                # Path to save the annotated media
)
```

### Example
```python
python examples/deepfake_detection/detection.py \
  --detection_model resnet_inception \
  --media_path assets/images/img1.jpg \
  --confidence_threshold 0.5
```  

### Output
```python
media_name | frame_number | is_deepfake | confidence | model_name
img1.jpg   |      0       |      False  |    0.99    | efficientnet
```

  
### Videos
```python
import torch
from mukh.deepfake_detection import DeepfakeDetector

# Initialize detector
detection_model = "efficientnet"                  # Other models "resnet_inception"

detector = DeepfakeDetector(
    model_name=detection_model,
    confidence_threshold=0.5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Detect deepfakes in Videos

media_path = "assets/videos/deepfake_elon_musk.mp4"             

detections, final_result = detector.detect(
    media_path=media_path,                                         # Path to the media file (image/video)
    save_csv=True,                                                 # Save the detections to a CSV file
    csv_path=f"output/{detection_model}/deepfake_detections.csv",  # Path to save the CSV file
    save_annotated=True,                                           # Save the annotated media
    output_folder=f"output/{detection_model}",                # Path to save the annotated media
    num_frames=11,                                                 # Number of equally spaced frames for video analysis
)

```

### Example
```python
python examples/deepfake_detection/detection.py \
  --detection_model resnet_inception \
  --media_path assets/videos/deepfake_elon_musk.mp4 \
  --confidence_threshold 0.5 \
  --num_frames 11
```

### Output
```python
      media_name       | frame_number |is_deepfake|confidence|      model_name
deepfake_elon_musk.mp4 |      0       |    True  |    0.99   |EfficientNetAutoAttB4
deepfake_elon_musk.mp4 |      43      |    True  |    0.69   |EfficientNetAutoAttB4
deepfake_elon_musk.mp4 |      86      |    False |    0.73   |EfficientNetAutoAttB4
deepfake_elon_musk.mp4 |     172      |    True  |    0.95   |EfficientNetAutoAttB4
deepfake_elon_musk.mp4 |     215      |    True  |    0.98   |EfficientNetAutoAttB4
deepfake_elon_musk.mp4 |     129      |    True  |    0.96   |EfficientNetAutoAttB4
deepfake_elon_musk.mp4 |     258      |    True  |    0.53   |EfficientNetAutoAttB4
deepfake_elon_musk.mp4 |     301      |    True  |    0.77   |EfficientNetAutoAttB4
deepfake_elon_musk.mp4 |     344      |    False |    0.83   |EfficientNetAutoAttB4
deepfake_elon_musk.mp4 |     387      |    True  |    0.62   |EfficientNetAutoAttB4
deepfake_elon_musk.mp4 |     431      |    False |    0.79   |EfficientNetAutoAttB4
```
```python
| deepfake_elon_musk.mp4 | EfficientNetAutoAttB4 | 8/11 deepfake frames | Final: DEEPFAKE
```
  
## Deepfake Detection Pipeline
  
```python
from mukh.pipelines.deepfake_detection import PipelineDeepfakeDetection

# Define model configurations with weights
model_configs = {
    "resnet_inception": 0.5,
    "efficientnet": 0.5
}

# Create ensemble detector
pipeline = PipelineDeepfakeDetection(model_configs)

media_path = "assets/videos/deepfake_elon_musk.mp4" # Or pass an image path

# Detect deepfakes
result = pipeline.detect(
    media_path=media_path,
    output_folder="output/deepfake_detection_pipeline",
    save_csv=True,
    num_frames=11,        # Number of equally spaced video frames for analysis
)
```

### Example
```python
python examples/pipelines/deepfake_detection.py \
  --media_path assets/videos/deepfake_elon_musk.mp4 \
  --output_folder output/deepfake_detection_pipeline
```

### Output
**Ensemble confidence score**
```python
frame_number|is_deepfake|confidence
     0      |   True    |   0.5
    43      |   True    |   0.84
    86      |   True    |   0.635
   129      |   True    |   0.98
   172      |   True    |   0.975
   215      |   True    |   0.99
   258      |   True    |   0.765
   301      |   True    |   0.885
   344      |   True    |   0.585
   387      |   True    |   0.81
   431      |   True    |   0.605
```

**Result from the respective models**
```python
 | deepfake_elon_musk.mp4 |   ResNetInception     | 10/11 deepfake frames | Final: DEEPFAKE
 | deepfake_elon_musk.mp4 | EfficientNetAutoAttB4 |  8/11 deepfake frames | Final: DEEPFAKE
```

**Final Pipeline Output**
```python
Final Ensemble Result: DEEPFAKE
Deepfake frames: 11/11
Average confidence: 0.7791
Model configurations: {
  'resnet_inception': 0.5,
  'efficientnet': 0.5
}
```
  
## Citations
For a detailed list of models, libraries, and datasets used, please refer to [CITATIONS.md](CITATIONS.md).
  
## Contact
For questions and feedback, please open an issue on GitHub.
