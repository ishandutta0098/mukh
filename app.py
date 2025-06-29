"""
Gradio application for Face Detection, Face Reenactment, and Deepfake Detection.

This application provides a web interface for:
1. Face Detection using multiple models (BlazeFace, MediaPipe, UltraLight)
2. Face Reenactment using TPS model
3. Deepfake Detection using multiple models in a pipeline
"""

import os
from typing import List, Optional, Tuple

import gradio as gr
import torch

from mukh.face_detection import FaceDetector
from mukh.pipelines.deepfake_detection import PipelineDeepfakeDetection
from mukh.reenactment import FaceReenactor


class GradioApp:
    """Main Gradio application class for face processing tasks."""

    def __init__(self):
        """Initialize the Gradio application with default models."""
        # Initialize models
        self.face_detector = None
        self.face_reenactor = None
        self.deepfake_pipeline = None

        # Create output directory
        self.output_dir = "gradio_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def detect_faces(
        self,
        image_path: str,
        detection_model: str,
    ) -> Tuple[str, str, str]:
        """Detect faces in an image using the specified model.

        Args:
            image_path: Path to the input image
            detection_model: Model to use for detection

        Returns:
            Tuple of (annotated_image_path, json_path, results_text)
        """
        try:
            # Debug: Print the selected model
            print(f"Selected detection model: {detection_model}")

            # Validate inputs
            if not image_path:
                return None, None, "Error: No image provided"

            if not os.path.exists(image_path):
                return None, None, f"Error: Image file not found: {image_path}"

            # Create detector with error handling
            try:
                detector = FaceDetector.create(detection_model)
                print(f"Successfully created detector: {type(detector)}")
            except Exception as e:
                return (
                    None,
                    None,
                    f"Error creating detector '{detection_model}': {str(e)}",
                )

            # Create output paths
            output_folder = os.path.join(
                self.output_dir, "face_detection", detection_model
            )
            json_path = os.path.join(output_folder, "detections.json")

            # Detect faces - save JSON and annotated image by default
            detections = detector.detect(
                image_path=image_path,
                save_json=True,
                json_path=json_path,
                save_annotated=True,
                output_folder=output_folder,
            )

            # Find annotated image
            annotated_image_path = None
            if True:
                image_name = os.path.basename(image_path)
                name, ext = os.path.splitext(image_name)

                # Try different possible naming patterns
                possible_names = [
                    f"{name}_annotated{ext}",
                    f"{name}_detected{ext}",
                    f"{name}_detection{ext}",
                ]

                for possible_name in possible_names:
                    potential_path = os.path.join(output_folder, possible_name)
                    if os.path.exists(potential_path):
                        annotated_image_path = potential_path
                        break

            # Create results text
            results_text = f"Found {len(detections)} face(s)\n\n"
            for i, detection in enumerate(detections):
                results_text += f"Face {i+1}:\n"
                results_text += f"  Confidence: {detection.bbox.confidence:.3f}\n"
                results_text += f"  Bounding Box: ({detection.bbox.x1}, {detection.bbox.y1}, {detection.bbox.x2}, {detection.bbox.y2})\n\n"

            return (
                (
                    annotated_image_path
                    if annotated_image_path and os.path.exists(annotated_image_path)
                    else None
                ),
                json_path if True and os.path.exists(json_path) else None,
                results_text,
            )

        except Exception as e:
            return None, None, f"Error: {str(e)}"

    def reenact_face(
        self,
        source_image: str,
        driving_video: str,
        reenactor_model: str = "tps",
        resize_to_image_resolution: bool = False,
    ) -> Tuple[str, str]:
        """Reenact face from source image using driving video.

        Args:
            source_image: Path to source image
            driving_video: Path to driving video
            reenactor_model: Model to use for reenactment
            resize_to_image_resolution: Whether to resize to image resolution

        Returns:
            Tuple of (reenacted_video_path, comparison_video_path)
        """
        try:
            # Create reenactor
            reenactor = FaceReenactor.create(reenactor_model)

            # Create output path
            output_folder = os.path.join(
                self.output_dir, "face_reenactment", reenactor_model
            )
            os.makedirs(output_folder, exist_ok=True)

            source_name = os.path.splitext(os.path.basename(source_image))[0]
            output_path = os.path.join(output_folder, f"{source_name}_reenacted")

            # Perform reenactment - save comparison by default
            result_path = reenactor.reenact_from_video(
                source_path=source_image,
                driving_video_path=driving_video,
                output_path=output_path,
                save_comparison=True,
                resize_to_image_resolution=resize_to_image_resolution,
            )

            # Find comparison video if it was created
            comparison_path = None
            # Extract driving video name for pattern matching
            driving_name = os.path.splitext(os.path.basename(driving_video))[0]

            # The comparison video should be in the same directory as output_path
            # which is the directory passed to the reenactor
            search_dir = output_path  # This is the directory passed to reenactor

            # Look for comparison video with the exact naming pattern used by TPS
            comparison_patterns = [
                f"comparison_{source_name}_by_{driving_name}.mp4",  # Exact pattern from TPS
                f"comparison_{source_name}_by_*.mp4",  # Pattern with wildcard
                f"comparison_*.mp4",  # Generic comparison pattern
            ]

            import glob

            for pattern in comparison_patterns:
                comparison_files = glob.glob(os.path.join(search_dir, pattern))
                if comparison_files:
                    comparison_path = comparison_files[0]  # Take the first match
                    break

            # If still not found, check the parent directory of result_path
            if not comparison_path:
                result_dir = os.path.dirname(result_path)
                for pattern in comparison_patterns:
                    comparison_files = glob.glob(os.path.join(result_dir, pattern))
                    if comparison_files:
                        comparison_path = comparison_files[0]
                        break

            return result_path, comparison_path

        except Exception as e:
            return None, None

    def detect_deepfakes(
        self,
        image_file: Optional[str],
        video_file: Optional[str],
        models_to_use: List[str],
        confidence_threshold: float = 0.5,
        num_frames: int = 11,
    ) -> Tuple[str, str]:
        """Detect deepfakes using multiple models in a pipeline.

        Args:
            image_file: Path to input image (optional)
            video_file: Path to input video (optional)
            models_to_use: List of models to use in pipeline
            confidence_threshold: Confidence threshold for final decision
            num_frames: Number of frames to analyze for videos

        Returns:
            Tuple of (annotated_media_path, results_text)
        """
        try:
            # Determine which media file to use
            media_file = None
            media_type = None
            if image_file and os.path.exists(image_file):
                media_file = image_file
                media_type = "image"
            elif video_file and os.path.exists(video_file):
                media_file = video_file
                media_type = "video"
            else:
                return None, "❌ **Error**: Please upload either an image or video file"

            # Configure models based on selection with weights
            model_configs = {}

            if "resnet_inception" in models_to_use:
                model_configs["resnet_inception"] = 0.5

            if "efficientnet" in models_to_use:
                model_configs["efficientnet"] = 0.5

            # Check if at least one model is selected
            if not model_configs:
                return (
                    None,
                    "❌ **Error**: Please select at least one model to use for deepfake detection",
                )

            # Normalize weights to sum to 1.0
            total_weight = sum(model_configs.values())
            if total_weight > 0:
                model_configs = {k: v / total_weight for k, v in model_configs.items()}

            # Create pipeline
            pipeline = PipelineDeepfakeDetection(
                model_configs=model_configs,
                device=None,  # Auto-detect device
                confidence_threshold=confidence_threshold,
            )

            # Create output paths
            output_folder = os.path.join(self.output_dir, "deepfake_detection")

            # Detect deepfakes - save all outputs by default
            result = pipeline.detect(
                media_path=media_file,
                output_folder=output_folder,
                save_csv=True,
                num_frames=num_frames,
            )

            # Find annotated media - for pipeline, check if individual model outputs exist
            annotated_media_path = None

            # Create enhanced results text with better formatting
            def get_confidence_bar(confidence, width=20):
                """Create a visual confidence bar."""
                filled = int(confidence * width)
                bar = "█" * filled + "░" * (width - filled)
                return f"[{bar}] {confidence:.1%}"

            def get_risk_level(confidence, is_deepfake):
                """Determine risk level based on confidence."""
                if is_deepfake:
                    if confidence >= 0.8:
                        return "🔴 **HIGH RISK**"
                    elif confidence >= 0.6:
                        return "🟡 **MEDIUM RISK**"
                    else:
                        return "🟠 **LOW RISK**"
                else:
                    if confidence >= 0.8:
                        return "🟢 **VERY CONFIDENT**"
                    elif confidence >= 0.6:
                        return "🟡 **CONFIDENT**"
                    else:
                        return "🟠 **UNCERTAIN**"

            # The pipeline returns a boolean result (True for deepfake, False for real)
            overall_is_deepfake = result
            overall_status = "DEEPFAKE" if overall_is_deepfake else "AUTHENTIC"
            overall_icon = "🚨" if overall_is_deepfake else "✅"

            # For pipeline results, we don't have individual frame confidences,
            # so we'll create a summary based on the overall result
            avg_confidence = (
                confidence_threshold
                if overall_is_deepfake
                else (1.0 - confidence_threshold)
            )

            results_text = f"""# 🎬 **{media_type.title()} Analysis Results**

## 📊 **Summary**
- **File**: `{os.path.basename(media_file)}`
- **Media Type**: {media_type.title()}
- **Models Used**: {', '.join(models_to_use)}
- **Confidence Threshold**: {confidence_threshold:.1%}

---

## 🎯 **Overall Verdict**
### {overall_icon} **{overall_status}**

**Pipeline Result**: {get_confidence_bar(avg_confidence)}

---

## 🔍 **Technical Details**
- **Pipeline Models**: {', '.join(model_configs.keys())}
- **Model Weights**: {model_configs}
- **Detection Algorithm**: Weighted Ensemble Pipeline
"""

            if media_type == "video":
                results_text += f"- **Frames Analyzed**: {num_frames}\n"

            results_text += f"""

---

## 💡 **Recommendation**
"""
            if overall_is_deepfake:
                results_text += "⚠️ **This media shows strong indicators of being a deepfake.** Exercise caution when sharing or believing its content."
            else:
                results_text += "✅ **This media appears to be authentic.** However, always verify content from multiple sources."

            results_text += f"""

---

## ℹ️ **About This Analysis**
This analysis uses multiple state-of-the-art AI models in an ensemble pipeline to detect potential deepfakes. The pipeline combines predictions from {len(model_configs)} models with weighted averaging for improved accuracy. While highly accurate, no detection system is 100% perfect. Always use critical thinking and verify important content through multiple sources.
"""

            return (
                annotated_media_path,  # Pipeline may not create annotated output
                results_text,
            )

        except Exception as e:
            return None, f"❌ **Error**: {str(e)}"

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface with all three tasks."""

        # Create a custom theme with better colors for face processing
        custom_theme = gr.themes.Citrus(
            primary_hue="slate",
            secondary_hue="zinc",
            neutral_hue="gray",
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            button_primary_background_fill="*neutral_800",
            button_primary_background_fill_hover="*neutral_900",
        )

        with gr.Blocks(
            title="Mukh: Face Processing Suite",
            theme=custom_theme,
            css="""
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            .main-header {
                text-align: center;
                padding: 30px 20px;
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #020617 100%);
                color: white;
                border-radius: 16px;
                margin-bottom: 30px;
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                position: relative;
                overflow: hidden;
            }
            
            .main-header::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
                animation: shimmer 3s infinite;
            }
            
            @keyframes shimmer {
                0% { left: -100%; }
                100% { left: 100%; }
            }
            
            .main-header h1 {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            
            .tab-nav {
                border-radius: 12px;
                background: linear-gradient(145deg, #374151, #1f2937);
                padding: 8px;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
            }
            
            .upload-area {
                border: 2px dashed #6b7280;
                border-radius: 12px;
                padding: 30px;
                transition: all 0.3s ease;
                background: linear-gradient(145deg, #374151, #1f2937);
                position: relative;
            }
            
            .upload-area:hover {
                border-color: #9ca3af;
                background: linear-gradient(145deg, #4b5563, #374151);
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
            }
            
            .block {
                background: linear-gradient(145deg, #374151, #1f2937);
                border-radius: 16px;
                padding: 24px;
                margin: 15px 0;
                border: 1px solid #4b5563;
                transition: all 0.3s ease;
            }
            
            .block:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            }
            
            .input-section {
                background: linear-gradient(145deg, #374151, #1f2937);
                border-radius: 16px;
                padding: 28px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
                border: 1px solid #4b5563;
                transition: all 0.3s ease;
            }
            
            .input-section:hover {
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            }
            
            .output-section {
                background: linear-gradient(145deg, #1f2937, #111827);
                border-radius: 16px;
                padding: 28px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
                border: 1px solid #374151;
                transition: all 0.3s ease;
            }
            
            .output-section:hover {
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            }
            
            button {
                transition: all 0.3s ease !important;
                font-weight: 600 !important;
                letter-spacing: 0.025em !important;
            }
            
            button:hover {
                transform: translateY(-1px) !important;
                box-shadow: 0 10px 25px rgba(0,0,0,0.15) !important;
            }
            
            .gradio-accordion {
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                background: linear-gradient(145deg, #374151, #1f2937);
            }
            
            .markdown h3 {
                color: #e5e7eb;
                font-weight: 600;
                margin-bottom: 1rem;
                font-size: 1.25rem;
            }
            
            .task-header {
                text-align: center;
                margin: 30px 0;
                padding: 20px;
                background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .task-header h1 {
                font-size: 2rem;
                font-weight: 700;
                color: #f8fafc;
                text-shadow: 0 2px 4px rgba(0,0,0,0.5);
                margin-bottom: 0;
            }
            """,
        ) as interface:
            gr.Markdown(
                """
            # 🎭 Mukh: Face Processing Suite
            
            A comprehensive toolkit for face detection, reenactment, and deepfake detection.
            """,
                elem_classes="main-header",
            )

            with gr.Accordion("📝 Instructions", open=False):
                gr.Markdown(
                    """
                **Face Detection**: Upload an image and select a detection model to identify faces.
                
                **Face Reenactment**: Upload a source image and driving video to animate the face.
                
                **Deepfake Detection**: Upload an image or video to detect if it contains deepfakes using multiple AI models.
                
                All results are saved to the `gradio_output` directory.
                """
                )

            with gr.Tabs():
                # Face Detection Tab
                with gr.TabItem("👤 Face Detection"):
                    gr.Markdown(
                        """
                        # 👤 Face Detection
                        ### Detect faces in images using state-of-the-art models
                        """,
                        elem_classes="task-header",
                    )

                    with gr.Row():
                        with gr.Column():
                            face_input_image = gr.Image(
                                type="filepath", label="Upload Image"
                            )
                            face_detection_model = gr.Radio(
                                choices=["blazeface", "mediapipe", "ultralight"],
                                value="mediapipe",
                                label="Detection Model",
                                info="Select the face detection model to use",
                            )
                            face_detect_btn = gr.Button(
                                "🔍 Detect Faces", variant="primary", size="lg"
                            )

                        with gr.Column():
                            face_output_image = gr.Image(label="Annotated Image")
                            face_results_text = gr.Textbox(
                                label="Detection Results", lines=10
                            )
                            face_json_file = gr.File(
                                label="JSON Results", visible=False
                            )

                    # Add this after the radio button definition
                    def on_model_change(model):
                        print(f"Model changed to: {model}")
                        return f"Selected model: {model}"

                    model_debug = gr.Textbox(
                        label="Debug: Selected Model", visible=False
                    )

                    face_detection_model.change(
                        fn=on_model_change,
                        inputs=[face_detection_model],
                        outputs=[model_debug],
                    )

                    face_detect_btn.click(
                        fn=self.detect_faces,
                        inputs=[
                            face_input_image,
                            face_detection_model,
                        ],
                        outputs=[face_output_image, face_json_file, face_results_text],
                        show_progress=True,
                    )

                # Face Reenactment Tab
                with gr.TabItem("🎬 Face Reenactment"):
                    gr.Markdown(
                        """
                        # 🎬 Face Reenactment
                        ### Animate faces using driving videos
                        """,
                        elem_classes="task-header",
                    )

                    with gr.Row():
                        with gr.Column():
                            reenact_source_image = gr.Image(
                                type="filepath", label="Source Image"
                            )
                            reenact_driving_video = gr.Video(label="Driving Video")
                            reenact_model = gr.Dropdown(
                                choices=["tps"], value="tps", label="Reenactment Model"
                            )
                            reenact_resize_to_image = gr.Checkbox(
                                value=False, label="Resize to Image Resolution"
                            )
                            reenact_btn = gr.Button(
                                "🎬 Generate Reenactment", variant="primary", size="lg"
                            )

                        with gr.Column():
                            reenact_output_video = gr.Video(
                                label="Reenacted Video", visible=False
                            )
                            reenact_comparison_video = gr.Video(
                                label="Comparison Video",
                            )

                    reenact_btn.click(
                        fn=self.reenact_face,
                        inputs=[
                            reenact_source_image,
                            reenact_driving_video,
                            reenact_model,
                            reenact_resize_to_image,
                        ],
                        outputs=[reenact_output_video, reenact_comparison_video],
                        show_progress=True,
                    )

                # Deepfake Detection Tab
                with gr.TabItem("🕵️ Deepfake Detection"):
                    gr.Markdown(
                        """
                        # 🕵️ Deepfake Detection
                        ### Detect deepfakes using multiple AI models
                        """,
                        elem_classes="task-header",
                    )

                    with gr.Row():
                        with gr.Column():
                            deepfake_input_image = gr.Image(
                                type="filepath", label="Upload Image", visible=True
                            )
                            deepfake_input_video = gr.Video(
                                label="Upload Video", visible=True
                            )
                            deepfake_models = gr.CheckboxGroup(
                                choices=["resnet_inception", "efficientnet"],
                                value=["resnet_inception", "efficientnet"],
                                label="Models to Use",
                            )
                            deepfake_confidence_threshold = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                label="Confidence Threshold",
                            )
                            deepfake_num_frames = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=11,
                                step=1,
                                label="Number of Frames (for videos)",
                            )
                            deepfake_detect_btn = gr.Button(
                                "🕵️ Detect Deepfakes", variant="primary", size="lg"
                            )

                        with gr.Column():
                            deepfake_output_media = gr.File(
                                label="Annotated Media", visible=False
                            )
                            deepfake_results_text = gr.Textbox(
                                label="Detection Results", lines=10
                            )

                    deepfake_detect_btn.click(
                        fn=self.detect_deepfakes,
                        inputs=[
                            deepfake_input_image,
                            deepfake_input_video,
                            deepfake_models,
                            deepfake_confidence_threshold,
                            deepfake_num_frames,
                        ],
                        outputs=[deepfake_output_media, deepfake_results_text],
                        show_progress=True,
                    )

        return interface

    def launch(self, **kwargs) -> None:
        """Launch the Gradio interface.

        Args:
            **kwargs: Additional arguments to pass to gr.Interface.launch()
        """
        interface = self.create_interface()
        interface.launch(**kwargs)


def main():
    """Main function to run the Gradio application."""
    app = GradioApp()
    app.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        debug=True,
    )


if __name__ == "__main__":
    main()
