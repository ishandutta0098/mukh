# Face Detection API

## FaceDetector

Factory class for creating face detection model instances.

### Methods

```python
create(model: str) -> BaseFaceDetector
```

Creates a face detector instance of the specified type.

**Parameters:**
- `model`: The type of detector to create ("blazeface", "mediapipe", "ultralight")

**Returns:**
- A BaseFaceDetector instance

```python
list_available_models() -> List[str]
```

Returns a list of available face detection model names.

**Returns:**
- List of supported model names

### BaseFaceDetector.detect()

```python
def detect(
    image_path: str,
    save_json: bool = False,
    json_path: str = "detections.json", 
    save_annotated: bool = False,
    output_folder: str = "output"
) -> List[FaceDetection]
```

**Parameters:**  
- `image_path`: Path to the input image  
- `save_json`: Whether to save detection results to JSON file  
- `json_path`: Path where to save the JSON file  
- `save_annotated`: Whether to save annotated image with bounding boxes  
- `output_folder`: Folder path where to save annotated images  

**Returns:**
- List of FaceDetection objects

## Available Models

- `blazeface`
- `mediapipe`
- `ultralight` 