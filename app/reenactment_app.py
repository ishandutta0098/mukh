"""
Minimal Face Reenactment App

A simple Gradio interface for face reenactment using TPS model.
"""

import os
from typing import Tuple

import gradio as gr
from config_loader import design_config

from mukh.reenactment import FaceReenactor


def reenact_face(
    source_image_path: str, driving_video_path: str, reenactment_model: str
) -> Tuple[str, str, str]:
    """Reenact face using source image and driving video.

    Args:
        source_image_path: Path to the source image
        driving_video_path: Path to the driving video
        reenactment_model: Model to use for reenactment

    Returns:
        Tuple of (output_video_path, comparison_video_path, results_text)
    """
    try:
        if not source_image_path or not os.path.exists(source_image_path):
            return None, None, "❌ Error: No valid source image provided"

        if not driving_video_path or not os.path.exists(driving_video_path):
            return None, None, "❌ Error: No valid driving video provided"

        # Create reenactor
        reenactor = FaceReenactor.create(reenactment_model)

        # Set output paths
        output_folder = os.path.join("output", "face_reenactment", reenactment_model)
        os.makedirs(output_folder, exist_ok=True)

        # Perform reenactment with comparison video
        output_video_path = reenactor.reenact_from_video(
            source_path=source_image_path,
            driving_video_path=driving_video_path,
            output_path=output_folder,
            save_comparison=True,
        )

        # Generate comparison video path
        source_name = os.path.splitext(os.path.basename(source_image_path))[0]
        driving_name = os.path.splitext(os.path.basename(driving_video_path))[0]
        comparison_video_path = os.path.join(
            output_folder, f"comparison_{source_name}_by_{driving_name}.mp4"
        )

        # Create detailed results text
        if output_video_path and os.path.exists(output_video_path):
            results_text = "✅ Reenactment Completed Successfully\n\n"
            results_text += f"🎬 Model Used: {reenactment_model.upper()}\n"
            results_text += f"📁 Output Location: {output_video_path}\n"
            results_text += f"📊 Source Image: {os.path.basename(source_image_path)}\n"
            results_text += (
                f"🎥 Driving Video: {os.path.basename(driving_video_path)}\n\n"
            )
            results_text += "🎯 Process Details:\n"
            results_text += "  • Motion extraction from driving video\n"
            results_text += "  • Face alignment and warping\n"
            results_text += "  • TPS (Thin Plate Spline) transformation\n"
            results_text += "  • High-quality video generation\n"
            results_text += "  • Side-by-side comparison video created\n\n"
            results_text += "💾 Output: Ready for download and viewing\n"
            results_text += f"🔄 Comparison Video: {comparison_video_path}"

            # Check if comparison video exists
            if not os.path.exists(comparison_video_path):
                comparison_video_path = None
        else:
            results_text = "❌ Error: Reenactment failed to generate output video"
            comparison_video_path = None

        return output_video_path, comparison_video_path, results_text

    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"


def create_interface():
    """Create the Gradio interface for face reenactment."""

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
        title="Face Reenactment - Mukh",
        theme=theme,
        css=design_config.get_css_styles(),
    ) as interface:

        # Modern Header Section
        with gr.Row():
            with gr.Column(elem_classes=["header-section"]):
                gr.HTML(
                    """
                    <div style="text-align: center;">
                        <h1>🎬 Mukh - Face Reenactment</h1>
                        <p>Animate faces in images using motion from driving videos with TPS technology</p>
                        <div style="display: flex; justify-content: center; gap: 25px; margin-top: 20px; flex-wrap: wrap;">
                            <div style="background: rgba(255,255,255,0.1); padding: 12px 18px; border-radius: 8px; backdrop-filter: blur(10px);">
                                <div style="font-size: 1.2em; margin-bottom: 5px;">🎯</div>
                                <strong>TPS Model</strong><br/>
                                <span style="opacity: 0.9;">Motion transfer</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 12px 18px; border-radius: 8px; backdrop-filter: blur(10px);">
                                <div style="font-size: 1.2em; margin-bottom: 5px;">🎨</div>
                                <strong>High Quality</strong><br/>
                                <span style="opacity: 0.9;">Professional output</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 12px 18px; border-radius: 8px; backdrop-filter: blur(10px);">
                                <div style="font-size: 1.2em; margin-bottom: 5px;">⚡</div>
                                <strong>Real-time</strong><br/>
                                <span style="opacity: 0.9;">Fast processing</span>
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

                source_image = gr.Image(
                    type="filepath",
                    label="📸 Source Image",
                    height=200,
                    elem_classes=["upload-area"],
                )

                driving_video = gr.Video(
                    label="🎥 Driving Video", height=200, elem_classes=["upload-area"]
                )

                gr.HTML(
                    """
                    <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 8px; padding: 15px; margin: 15px 0;">
                        <div style="color: #667eea; font-weight: 600; margin-bottom: 8px;">📋 Input Requirements:</div>
                        <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.4;">
                            <strong>Source Image:</strong> JPG, PNG, WEBP (face clearly visible)<br/>
                            <strong>Driving Video:</strong> MP4, AVI, MOV (with facial expressions)
                        </div>
                    </div>
                    """
                )

                reenactment_model = gr.Radio(
                    choices=["tps"],
                    value="tps",
                    label="🤖 Reenactment Model",
                    info="TPS (Thin Plate Spline) motion transfer",
                )

                gr.HTML(
                    """
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 15px; margin: 15px 0;">
                        <div style="color: #10b981; font-weight: 600; margin-bottom: 8px;">💡 How TPS Works:</div>
                        <div style="color: #a7f3d0; font-size: 0.85rem; line-height: 1.4;">
                            <strong>Step 1:</strong> Extract facial keypoints from both images<br/>
                            <strong>Step 2:</strong> Calculate motion vectors from driving video<br/>
                            <strong>Step 3:</strong> Apply TPS transformation to source face<br/>
                            <strong>Step 4:</strong> Generate smooth, realistic animation
                        </div>
                    </div>
                    """
                )

                reenact_btn = gr.Button(
                    "🎬 Generate Reenactment",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-button"],
                )

            # Results Section
            with gr.Column(scale=1, elem_classes=["results-section"]):
                gr.HTML(
                    '<h3 style="color: #667eea; font-weight: bold; border-bottom: 2px solid #667eea; padding-bottom: 5px; margin-bottom: 20px;">🎥 Reenactment Results</h3>'
                )

                output_video = gr.Video(
                    label="🎬 Generated Video",
                    height=250,
                    elem_classes=["output-video"],
                )

                comparison_video = gr.Video(
                    label="🔄 Comparison Video (Source | Driving | Generated)",
                    height=200,
                    elem_classes=["output-video"],
                )

        # Information Section
        with gr.Row():
            with gr.Column(elem_classes=["gallery-section"]):
                gr.HTML(
                    '<h3 style="color: #667eea; font-weight: bold; border-bottom: 2px solid #667eea; padding-bottom: 5px; margin-bottom: 20px;">ℹ️ How Face Reenactment Works</h3>'
                )

                gr.HTML(
                    """
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px; text-align: center;">
                            <div style="font-size: 2em; margin-bottom: 10px;">📸</div>
                            <strong style="color: #e2e8f0;">1. Upload Source</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">Provide a clear image with a visible face</p>
                        </div>
                        
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px; text-align: center;">
                            <div style="font-size: 2em; margin-bottom: 10px;">🎥</div>
                            <strong style="color: #e2e8f0;">2. Add Driving Video</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">Upload video with facial expressions</p>
                        </div>
                        
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px; text-align: center;">
                            <div style="font-size: 2em; margin-bottom: 10px;">🔄</div>
                            <strong style="color: #e2e8f0;">3. Process Motion</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">AI extracts and transfers motion</p>
                        </div>
                        
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px; text-align: center;">
                            <div style="font-size: 2em; margin-bottom: 10px;">🎬</div>
                            <strong style="color: #e2e8f0;">4. Generate Videos</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">Create animated face + comparison video</p>
                        </div>
                    </div>
                    
                    <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 10px; padding: 20px; margin-top: 20px;">
                        <div style="color: #f59e0b; font-weight: 600; margin-bottom: 15px; font-size: 1.1rem;">⚡ Performance Tips:</div>
                        <div style="color: #fcd34d; font-size: 0.9rem; line-height: 1.6;">
                            • Use high-quality source images with clear, well-lit faces<br/>
                            • Driving videos should have diverse facial expressions<br/>
                            • Keep videos under 30 seconds for faster processing<br/>
                            • Ensure both source and driving content have similar lighting<br/>
                            • The comparison video shows: Source | Driving | Generated frames side-by-side
                        </div>
                    </div>
                    """
                )

        # Event handlers
        reenact_btn.click(
            fn=reenact_face,
            inputs=[source_image, driving_video, reenactment_model],
            outputs=[output_video, comparison_video],
            show_progress=True,
        )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        show_api=False,
    )
