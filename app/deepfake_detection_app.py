"""
Minimal Deepfake Detection App

A simple Gradio interface for deepfake detection using pipeline approach.
"""

import os
from typing import List

import gradio as gr
from config_loader import design_config

from mukh.pipelines.deepfake_detection import PipelineDeepfakeDetection


def detect_deepfakes(
    media_path: str,
    detection_method: str,
    selected_models: List[str],
    resnet_weight: float,
    efficientnet_weight: float,
) -> str:
    """Detect deepfakes in an image or video using pipeline approach.

    Args:
        media_path: Path to the input media file
        detection_method: Method to use ('ensemble' or 'individual')
        selected_models: List of selected models for individual detection
        resnet_weight: Weight for ResNet Inception model in ensemble
        efficientnet_weight: Weight for EfficientNet model in ensemble

    Returns:
        String containing the results text
    """
    try:
        if not media_path or not os.path.exists(media_path):
            return "❌ Error: No valid media file provided"

        # Set output folder
        output_folder = os.path.join("output", "deepfake_detection")
        os.makedirs(output_folder, exist_ok=True)

        if detection_method == "ensemble":
            # Check if weights sum to 1.0
            total_weight = resnet_weight + efficientnet_weight
            if abs(total_weight - 1.0) > 0.001:  # Allow small floating point tolerance
                return f"⚠️ Warning: Model weights must sum to 1.0. Current sum: {total_weight:.3f}\nPlease adjust the weights so ResNet + EfficientNet = 1.0"

            model_configs = {
                "resnet_inception": resnet_weight,
                "efficientnet": efficientnet_weight,
            }

            detector = PipelineDeepfakeDetection(
                model_configs=model_configs, device=None, confidence_threshold=0.5
            )

            # Perform detection
            result = detector.detect(
                media_path=media_path,
                output_folder=output_folder,
                save_csv=True,
                num_frames=11,
            )

        else:
            # Individual model detection
            if not selected_models:
                return "❌ Error: Please select at least one model for individual detection"

            for model_name in selected_models:
                # Create single model detector
                model_configs = {model_name: 1.0}
                detector = PipelineDeepfakeDetection(
                    model_configs=model_configs, device=None, confidence_threshold=0.5
                )

                result = detector.detect(
                    media_path=media_path,
                    output_folder=output_folder,
                    save_csv=True,
                    num_frames=11,
                )

        # Read pipeline results if available
        pipeline_result_path = os.path.join(output_folder, "pipeline_result.txt")

        if os.path.exists(pipeline_result_path):
            with open(pipeline_result_path, "r") as f:
                pipeline_content = f.read().strip()
            return pipeline_content
        else:
            return "❌ Error: Pipeline results file not found"

    except Exception as e:
        return f"❌ Error: {str(e)}"


def create_interface():
    """Create the Gradio interface for deepfake detection."""

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
        title="Deepfake Detection - Mukh",
        theme=theme,
        css=design_config.get_css_styles(),
    ) as interface:

        # Modern Header Section
        with gr.Row():
            with gr.Column(elem_classes=["header-section"]):
                gr.HTML(
                    """
                    <div style="text-align: center;">
                        <h1>🕵️ Mukh - Deepfake Detection</h1>
                        <p>Detect artificially generated or manipulated faces using ensemble AI models</p>
                        <div style="display: flex; justify-content: center; gap: 25px; margin-top: 20px; flex-wrap: wrap;">
                            <div style="background: rgba(255,255,255,0.1); padding: 12px 18px; border-radius: 8px; backdrop-filter: blur(10px);">
                                <div style="font-size: 1.2em; margin-bottom: 5px;">🧠</div>
                                <strong>ResNet Inception</strong><br/>
                                <span style="opacity: 0.9;">Deep analysis</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 12px 18px; border-radius: 8px; backdrop-filter: blur(10px);">
                                <div style="font-size: 1.2em; margin-bottom: 5px;">⚡</div>
                                <strong>EfficientNet</strong><br/>
                                <span style="opacity: 0.9;">Fast detection</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 12px 18px; border-radius: 8px; backdrop-filter: blur(10px);">
                                <div style="font-size: 1.2em; margin-bottom: 5px;">🎯</div>
                                <strong>Ensemble</strong><br/>
                                <span style="opacity: 0.9;">High accuracy</span>
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

                input_media = gr.File(
                    label="📁 Upload Media File",
                    file_types=["image", "video"],
                    elem_classes=["upload-area"],
                )

                gr.HTML(
                    """
                    <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 8px; padding: 15px; margin: 15px 0;">
                        <div style="color: #667eea; font-weight: 600; margin-bottom: 8px;">📋 Supported Formats:</div>
                        <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.4;">
                            <strong>Images:</strong> JPG, JPEG, PNG, BMP, TIFF, WEBP<br/>
                            <strong>Videos:</strong> MP4, AVI, MOV, MKV, FLV, WMV
                        </div>
                    </div>
                    """
                )

                # Detection Method Selection
                detection_method = gr.Radio(
                    choices=["ensemble", "individual"],
                    value="ensemble",
                    label="🤖 Detection Method",
                    info="Choose between ensemble (recommended) or individual model analysis",
                )

                # Model Selection for Individual Analysis (hidden by default)
                with gr.Group(visible=False) as individual_group:
                    selected_models = gr.CheckboxGroup(
                        choices=["resnet_inception", "efficientnet"],
                        value=["resnet_inception"],
                        label="🔧 Select Models",
                        info="Choose which models to use for individual analysis",
                    )

                # Ensemble Weights (visible by default)
                with gr.Group(visible=True) as ensemble_group:
                    gr.HTML(
                        """
                        <div style="color: #10b981; font-weight: 600; margin-bottom: 10px;">⚖️ Model Weights (Ensemble):</div>
                        """
                    )
                    resnet_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="🧠 ResNet Inception Weight",
                        info="Weight for ResNet Inception model in ensemble",
                    )
                    efficientnet_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="⚡ EfficientNet Weight",
                        info="Weight for EfficientNet model in ensemble",
                    )

                gr.HTML(
                    """
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 15px; margin: 15px 0;">
                        <div style="color: #10b981; font-weight: 600; margin-bottom: 8px;">🤖 Detection Methods:</div>
                        <div style="color: #a7f3d0; font-size: 0.85rem; line-height: 1.4;">
                            <strong>Ensemble:</strong> Combines both models with custom weights for best accuracy<br/>
                            <strong>Individual:</strong> Analyze with selected models separately for comparison<br/>
                            <strong>Weights:</strong> Higher weight = more influence on final decision
                        </div>
                    </div>
                    """
                )

                detect_btn = gr.Button(
                    "🕵️ Detect Deepfakes",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-button"],
                )

                # Function to toggle visibility based on detection method
                def toggle_method_options(method):
                    if method == "ensemble":
                        return gr.Group(visible=False), gr.Group(visible=True)
                    else:
                        return gr.Group(visible=True), gr.Group(visible=False)

                detection_method.change(
                    fn=toggle_method_options,
                    inputs=[detection_method],
                    outputs=[individual_group, ensemble_group],
                )

            # Results Section
            with gr.Column(scale=1, elem_classes=["results-section"]):
                gr.HTML(
                    '<h3 style="color: #667eea; font-weight: bold; border-bottom: 2px solid #667eea; padding-bottom: 5px; margin-bottom: 20px;">📊 Detection Results</h3>'
                )

                results_text = gr.Textbox(
                    label="📋 Pipeline Analysis Report",
                    lines=15,
                    max_lines=30,
                    elem_classes=["output-text"],
                    show_copy_button=True,
                    placeholder="Upload an image or video file, configure detection settings, and click 'Detect Deepfakes' to see detailed analysis results here...",
                )

        # Information Section
        with gr.Row():
            with gr.Column(elem_classes=["gallery-section"]):
                gr.HTML(
                    '<h3 style="color: #667eea; font-weight: bold; border-bottom: 2px solid #667eea; padding-bottom: 5px; margin-bottom: 20px;">ℹ️ How Deepfake Detection Works</h3>'
                )

                gr.HTML(
                    """
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px; text-align: center;">
                            <div style="font-size: 2em; margin-bottom: 10px;">📤</div>
                            <strong style="color: #e2e8f0;">1. Upload Media</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">Select image or video file</p>
                        </div>
                        
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px; text-align: center;">
                            <div style="font-size: 2em; margin-bottom: 10px;">🤖</div>
                            <strong style="color: #e2e8f0;">2. Choose Method</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">Ensemble or individual models</p>
                        </div>
                        
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px; text-align: center;">
                            <div style="font-size: 2em; margin-bottom: 10px;">⚖️</div>
                            <strong style="color: #e2e8f0;">3. Set Weights</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">Configure model influence</p>
                        </div>
                        
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px; text-align: center;">
                            <div style="font-size: 2em; margin-bottom: 10px;">📊</div>
                            <strong style="color: #e2e8f0;">4. Get Results</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">Confidence scores and analysis</p>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                        <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid #8b5cf6; border-radius: 10px; padding: 20px;">
                            <div style="color: #8b5cf6; font-weight: 600; margin-bottom: 15px; font-size: 1.1rem;">🧠 Detection Technology:</div>
                            <div style="color: #c4b5fd; font-size: 0.9rem; line-height: 1.6;">
                                • <strong>ResNet Inception:</strong> Advanced convolutional neural network for deep feature extraction<br/>
                                • <strong>EfficientNet:</strong> Optimized architecture balancing accuracy and efficiency<br/>
                                • <strong>Ensemble Learning:</strong> Combines multiple models with weighted averaging for improved reliability
                            </div>
                        </div>
                        
                        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 10px; padding: 20px;">
                            <div style="color: #10b981; font-weight: 600; margin-bottom: 15px; font-size: 1.1rem;">⚖️ Weight Configuration:</div>
                            <div style="color: #a7f3d0; font-size: 0.9rem; line-height: 1.6;">
                                • <strong>Equal Weights (0.5, 0.5):</strong> Balanced approach, recommended for most cases<br/>
                                • <strong>ResNet Heavy (0.7, 0.3):</strong> Emphasizes deep feature analysis<br/>
                                • <strong>EfficientNet Heavy (0.3, 0.7):</strong> Prioritizes efficiency and speed<br/>
                                • <strong>Custom Weights:</strong> Fine-tune based on your specific needs
                            </div>
                        </div>
                    </div>
                    
                    <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 10px; padding: 20px; margin-top: 20px;">
                        <div style="color: #f59e0b; font-weight: 600; margin-bottom: 15px; font-size: 1.1rem;">⚡ Performance Tips:</div>
                        <div style="color: #fcd34d; font-size: 0.9rem; line-height: 1.6;">
                            • Use ensemble method with equal weights (0.5, 0.5) for best overall accuracy<br/>
                            • Individual analysis is useful for understanding model-specific strengths<br/>
                            • Higher resolution media generally provides better detection accuracy<br/>
                            • For videos, ensure faces are clearly visible throughout the clip
                        </div>
                    </div>
                    
                    <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 10px; padding: 20px; margin-top: 20px;">
                        <div style="color: #ef4444; font-weight: 600; margin-bottom: 15px; font-size: 1.1rem;">⚠️ Important Considerations:</div>
                        <div style="color: #fca5a5; font-size: 0.9rem; line-height: 1.6;">
                            • This tool is for educational and research purposes<br/>
                            • Results should be interpreted by qualified professionals<br/>
                            • False positives/negatives are possible with any AI system<br/>
                            • Always consider ethical implications when using deepfake detection
                        </div>
                    </div>
                    """
                )

        # Event handlers
        detect_btn.click(
            fn=detect_deepfakes,
            inputs=[
                input_media,
                detection_method,
                selected_models,
                resnet_weight,
                efficientnet_weight,
            ],
            outputs=[results_text],
            show_progress=True,
        )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_api=False,
    )
