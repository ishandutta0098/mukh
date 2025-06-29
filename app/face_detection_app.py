"""
Minimal Face Detection App

A simple Gradio interface for face detection using multiple models.
"""

import os
from typing import Tuple

import gradio as gr
from config_loader import design_config

from mukh.face_detection import FaceDetector


def detect_faces(image_path: str, detection_model: str) -> Tuple[str, str, str]:
    """Detect faces in an image using the specified model.

    Args:
        image_path: Path to the input image
        detection_model: Model to use for detection

    Returns:
        Tuple of (annotated_image_path, json_path, results_text)
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return None, None, "❌ Error: No valid image provided"

        # Create detector
        detector = FaceDetector.create(detection_model)

        # Set output paths
        output_folder = os.path.join("output", "face_detection", detection_model)
        json_path = os.path.join(output_folder, "detections.json")

        # Detect faces
        detections = detector.detect(
            image_path=image_path,
            save_json=True,
            json_path=json_path,
            save_annotated=True,
            output_folder=output_folder,
        )

        # Find annotated image - face detectors save with '_detected' suffix
        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        annotated_path = os.path.join(output_folder, f"{name}_detected{ext}")

        # Create detailed results text
        results_text = f"✅ Detection Completed Successfully\n\n"
        results_text += f"🔍 Model Used: {detection_model.title()}\n"
        results_text += f"📊 Faces Found: {len(detections)}\n\n"

        if detections:
            results_text += "📋 Detection Details:\n"
            for i, detection in enumerate(detections, 1):
                results_text += f"\nFace {i}:\n"
                results_text += f"  • Confidence: {detection.bbox.confidence:.3f}\n"
                results_text += f"  • Coordinates: ({detection.bbox.x1}, {detection.bbox.y1}) → ({detection.bbox.x2}, {detection.bbox.y2})\n"
                results_text += f"  • Size: {detection.bbox.x2 - detection.bbox.x1} × {detection.bbox.y2 - detection.bbox.y1} pixels\n"
        else:
            results_text += "ℹ️ No faces were detected in the image."

        return (
            annotated_path if os.path.exists(annotated_path) else None,
            json_path if os.path.exists(json_path) else None,
            results_text,
        )

    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"


def create_interface():
    """Create the Gradio interface for face detection."""

    # Get theme configuration from design config
    theme_config = design_config.get_theme_config()
    theme_colors = design_config.get_theme_colors()

    # Create theme with configuration
    theme = gr.themes.Soft(
        primary_hue=theme_config["primary_hue"],
        secondary_hue=theme_config["secondary_hue"],
        neutral_hue=theme_config["neutral_hue"],
    ).set(
        body_background_fill_dark=theme_colors["body_background_fill_dark"],
        background_fill_primary_dark=theme_colors["background_fill_primary_dark"],
        background_fill_secondary_dark=theme_colors["background_fill_secondary_dark"],
        border_color_primary_dark=theme_colors["border_color_primary_dark"],
    )

    with gr.Blocks(
        title="Face Detection - Mukh",
        theme=theme,
        css=design_config.get_css_styles(),
    ) as interface:

        # Modern Header Section
        with gr.Row():
            with gr.Column(elem_classes=["header-section"]):
                gr.HTML(
                    """
                    <div style="text-align: center;">
                        <h1>👤 Mukh - Face Detection</h1>
                        <p>Detect and analyze faces in images using state-of-the-art AI models</p>
                        <div style="display: flex; justify-content: center; gap: 25px; margin-top: 20px; flex-wrap: wrap;">
                            <div style="background: rgba(255,255,255,0.1); padding: 12px 18px; border-radius: 8px; backdrop-filter: blur(10px);">
                                <div style="font-size: 1.2em; margin-bottom: 5px;">🔥</div>
                                <strong>BlazeFace</strong><br/>
                                <span style="opacity: 0.9;">Fast & lightweight</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 12px 18px; border-radius: 8px; backdrop-filter: blur(10px);">
                                <div style="font-size: 1.2em; margin-bottom: 5px;">🎯</div>
                                <strong>MediaPipe</strong><br/>
                                <span style="opacity: 0.9;">High accuracy</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 12px 18px; border-radius: 8px; backdrop-filter: blur(10px);">
                                <div style="font-size: 1.2em; margin-bottom: 5px;">⚡</div>
                                <strong>UltraLight</strong><br/>
                                <span style="opacity: 0.9;">Ultra fast</span>
                            </div>
                        </div>
                    </div>
                    """,
                )

        with gr.Row(equal_height=True):
            # Input Section
            with gr.Column(scale=1, elem_classes=["search-section"]):
                gr.HTML(
                    '<h3 style="color: #667eea; font-weight: bold; border-bottom: 2px solid #667eea; padding-bottom: 5px; margin-bottom: 20px;">📤 Input Configuration</h3>'
                )

                input_image = gr.Image(
                    type="filepath",
                    label="Upload Image",
                    height=300,
                    elem_classes=["upload-area"],
                )

                gr.HTML(
                    """
                    <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 8px; padding: 15px; margin: 15px 0;">
                        <div style="color: #667eea; font-weight: 600; margin-bottom: 8px;">📋 Supported Formats:</div>
                        <div style="color: #cbd5e1; font-size: 0.9rem;">
                            JPG, JPEG, PNG, BMP, TIFF, WEBP
                        </div>
                    </div>
                    """
                )

                detection_model = gr.Radio(
                    choices=["blazeface", "mediapipe", "ultralight"],
                    value="mediapipe",
                    label="🤖 Detection Model",
                    info="Choose the AI model for face detection",
                )

                gr.HTML(
                    """
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 15px; margin: 15px 0;">
                        <div style="color: #10b981; font-weight: 600; margin-bottom: 8px;">💡 Model Comparison:</div>
                        <div style="color: #a7f3d0; font-size: 0.85rem; line-height: 1.4;">
                            <strong>BlazeFace:</strong> Mobile-optimized, fastest processing<br/>
                            <strong>MediaPipe:</strong> Best balance of speed and accuracy<br/>
                            <strong>UltraLight:</strong> Minimal computational requirements
                        </div>
                    </div>
                    """
                )

                detect_btn = gr.Button(
                    "🔍 Detect Faces",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-button"],
                )

            # Results Section
            with gr.Column(scale=1, elem_classes=["results-section"]):
                gr.HTML(
                    '<h3 style="color: #667eea; font-weight: bold; border-bottom: 2px solid #667eea; padding-bottom: 5px; margin-bottom: 20px;">📊 Detection Results</h3>'
                )

                output_image = gr.Image(
                    label="🖼️ Annotated Image", height=300, elem_classes=["output-image"]
                )

                results_text = gr.Textbox(
                    label="📋 Detection Analysis",
                    lines=8,
                    max_lines=15,
                    elem_classes=["output-text"],
                    show_copy_button=True,
                    placeholder="Upload an image and click 'Detect Faces' to see results here...",
                )

                json_file = gr.File(
                    label="📄 Download JSON Results",
                    visible=True,
                    file_types=[".json"],
                    elem_classes=["download-section"],
                )

        # Information Section
        with gr.Row():
            with gr.Column(elem_classes=["gallery-section"]):
                gr.HTML(
                    '<h3 style="color: #667eea; font-weight: bold; border-bottom: 2px solid #667eea; padding-bottom: 5px; margin-bottom: 20px;">ℹ️ How It Works</h3>'
                )

                gr.HTML(
                    """
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px; text-align: center;">
                            <div style="font-size: 2em; margin-bottom: 10px;">📤</div>
                            <strong style="color: #e2e8f0;">1. Upload Image</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">Select and upload your image file</p>
                        </div>
                        
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px; text-align: center;">
                            <div style="font-size: 2em; margin-bottom: 10px;">🤖</div>
                            <strong style="color: #e2e8f0;">2. Choose Model</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">Select detection algorithm</p>
                        </div>
                        
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px; text-align: center;">
                            <div style="font-size: 2em; margin-bottom: 10px;">🔍</div>
                            <strong style="color: #e2e8f0;">3. Detect Faces</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">AI analyzes and locates faces</p>
                        </div>
                        
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px; text-align: center;">
                            <div style="font-size: 2em; margin-bottom: 10px;">📊</div>
                            <strong style="color: #e2e8f0;">4. View Results</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">Get annotated image and data</p>
                        </div>
                    </div>
                    """
                )

        # Event handlers
        detect_btn.click(
            fn=detect_faces,
            inputs=[input_image, detection_model],
            outputs=[output_image, json_file, results_text],
            show_progress=True,
        )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
    )
