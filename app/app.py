"""
Mukh Apps Central Hub

A central application that provides access to all Mukh face processing applications
with descriptions and direct links to launch each app.
"""

import webbrowser

import gradio as gr
from config_loader import design_config


def launch_app(app_name: str, port: int) -> str:
    """Launch a specific app and return launch message.

    Args:
        app_name: Name of the app to launch
        port: Port number for the app

    Returns:
        Launch confirmation message
    """
    try:
        # Open the app in browser
        url = f"http://localhost:{port}"
        webbrowser.open(url)
        return f"✅ {app_name} launched! Opening {url} in your browser..."
    except Exception as e:
        return f"❌ Error launching {app_name}: {str(e)}\n\nPlease manually run: python {app_name.lower().replace(' ', '_')}_app.py"


def create_interface():
    """Create the central hub interface."""

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
        title="Mukh Apps - Central Hub",
        theme=theme,
        css=design_config.get_css_styles(),
    ) as interface:

        # Modern Header Section
        with gr.Row():
            with gr.Column(elem_classes=["header-section"]):
                gr.HTML(
                    """
                    <div style="text-align: center;">
                        <h1>🎭 Mukh - Fast and Comprehensive Face Analysis Suite</h1>
                        <p>Advanced AI-powered face detection, reenactment, and deepfake detection applications</p>
                        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 25px; flex-wrap: wrap;">
                            <div style="background: rgba(255,255,255,0.1); padding: 15px 20px; border-radius: 10px; backdrop-filter: blur(10px);">
                                <div style="font-size: 1.5em; margin-bottom: 5px;">👤</div>
                                <strong>Face Detection</strong><br/>
                                <span style="opacity: 0.9;">Multi-model detection</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 15px 20px; border-radius: 10px; backdrop-filter: blur(10px);">
                                <div style="font-size: 1.5em; margin-bottom: 5px;">🎬</div>
                                <strong>Face Reenactment</strong><br/>
                                <span style="opacity: 0.9;">TPS motion transfer</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 15px 20px; border-radius: 10px; backdrop-filter: blur(10px);">
                                <div style="font-size: 1.5em; margin-bottom: 5px;">🕵️</div>
                                <strong>Deepfake Detection</strong><br/>
                                <span style="opacity: 0.9;">Ensemble AI models</span>
                            </div>
                        </div>
                    </div>
                    """,
                )

        # Status display for launch messages
        status_display = gr.Textbox(
            label="🚀 Launch Status",
            value="Ready to launch applications...",
            interactive=False,
            elem_classes="status-message",
        )

        # Application Cards Section
        gr.HTML(
            '<h2 style="text-align: center; color: #f8fafc; font-size: 2rem; font-weight: 700; margin: 40px 0 30px 0; text-shadow: 2px 2px 8px rgba(0,0,0,0.4);">🚀 Available Applications</h2>'
        )

        with gr.Row(equal_height=True):
            # Face Detection App Card
            with gr.Column(scale=1):
                with gr.Group(elem_classes="app-card"):
                    gr.HTML(
                        '<div style="display: flex; align-items: center; margin-bottom: 15px;"><span style="font-size: 2em; margin-right: 10px;">👤</span><h3 style="color: #f8fafc; font-size: 1.5rem; font-weight: 600; margin: 0;">Face Detection</h3></div>'
                    )

                    gr.Markdown(
                        """
                        **Detect and analyze faces in images using state-of-the-art AI models.**
                        
                        Upload any image and get precise face locations with confidence scores and detailed analysis.
                        """,
                        elem_classes="app-description",
                    )

                    gr.HTML(
                        """
                        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 15px; margin: 15px 0;">
                            <div style="color: #10b981; font-weight: 600; margin-bottom: 8px;">✨ Key Features:</div>
                            <div style="color: #a7f3d0; font-size: 0.9rem; line-height: 1.5;">
                                • Multiple detection models (BlazeFace, MediaPipe, UltraLight)<br/>
                                • Bounding box coordinates and confidence scores<br/>
                                • JSON export of detection results<br/>
                                • Annotated image output with face highlights
                            </div>
                        </div>
                        """
                    )

                    face_detection_btn = gr.Button(
                        "🚀 Launch Face Detection",
                        variant="primary",
                        size="lg",
                        elem_classes="primary-button",
                    )

            # Face Reenactment App Card
            with gr.Column(scale=1):
                with gr.Group(elem_classes="app-card"):
                    gr.HTML(
                        '<div style="display: flex; align-items: center; margin-bottom: 15px;"><span style="font-size: 2em; margin-right: 10px;">🎬</span><h3 style="color: #f8fafc; font-size: 1.5rem; font-weight: 600; margin: 0;">Face Reenactment</h3></div>'
                    )

                    gr.Markdown(
                        """
                        **Animate faces in images using motion from driving videos.**
                        
                        Create realistic face animations by transferring expressions and movements with advanced TPS models.
                        """,
                        elem_classes="app-description",
                    )

                    gr.HTML(
                        """
                        <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid #3b82f6; border-radius: 8px; padding: 15px; margin: 15px 0;">
                            <div style="color: #3b82f6; font-weight: 600; margin-bottom: 8px;">✨ Key Features:</div>
                            <div style="color: #93c5fd; font-size: 0.9rem; line-height: 1.5;">
                                • TPS (Thin Plate Spline) motion transfer<br/>
                                • Source image + driving video input<br/>
                                • High-quality video output<br/>
                                • Side-by-side comparison videos
                            </div>
                        </div>
                        """
                    )

                    reenactment_btn = gr.Button(
                        "🚀 Launch Face Reenactment",
                        variant="primary",
                        size="lg",
                        elem_classes="primary-button",
                    )

            # Deepfake Detection App Card
            with gr.Column(scale=1):
                with gr.Group(elem_classes="app-card"):
                    gr.HTML(
                        '<div style="display: flex; align-items: center; margin-bottom: 15px;"><span style="font-size: 2em; margin-right: 10px;">🕵️</span><h3 style="color: #f8fafc; font-size: 1.5rem; font-weight: 600; margin: 0;">Deepfake Detection</h3></div>'
                    )

                    gr.Markdown(
                        """
                        **Detect artificially generated or manipulated faces in images and videos.**
                        
                        Uses multiple AI models in an ensemble pipeline for improved accuracy and reliable detection.
                        """,
                        elem_classes="app-description",
                    )

                    gr.HTML(
                        """
                        <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid #8b5cf6; border-radius: 8px; padding: 15px; margin: 15px 0;">
                            <div style="color: #8b5cf6; font-weight: 600; margin-bottom: 8px;">✨ Key Features:</div>
                            <div style="color: #c4b5fd; font-size: 0.9rem; line-height: 1.5;">
                                • Multi-model ensemble detection<br/>
                                • Support for images and videos<br/>
                                • ResNet Inception and EfficientNet models<br/>
                                • Confidence scoring and detailed analysis
                            </div>
                        </div>
                        """
                    )

                    deepfake_btn = gr.Button(
                        "🚀 Launch Deepfake Detection",
                        variant="primary",
                        size="lg",
                        elem_classes="primary-button",
                    )

        # Information Section
        with gr.Row():
            with gr.Column(elem_classes="search-section"):
                gr.HTML(
                    '<h3 style="color: #667eea; font-weight: bold; border-bottom: 2px solid #667eea; padding-bottom: 5px; margin-bottom: 20px;">💡 Getting Started</h3>'
                )

                gr.HTML(
                    """
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px;">
                            <div style="font-size: 1.2em; margin-bottom: 10px;">🎯 Step 1</div>
                            <strong style="color: #e2e8f0;">Choose Your Application</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">Click any launch button above to open the desired application in a new browser tab.</p>
                        </div>
                        
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px;">
                            <div style="font-size: 1.2em; margin-bottom: 10px;">📤 Step 2</div>
                            <strong style="color: #e2e8f0;">Upload Your Media</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">Each app accepts different file types. Upload your images or videos and configure the settings.</p>
                        </div>
                        
                        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; border-radius: 10px; padding: 20px;">
                            <div style="font-size: 1.2em; margin-bottom: 10px;">💾 Step 3</div>
                            <strong style="color: #e2e8f0;">Download Results</strong>
                            <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.9rem;">All results are automatically saved to the output/ directory and available for download.</p>
                        </div>
                    </div>
                    """
                )

        # Technical Information Section
        with gr.Row():
            with gr.Column(scale=1, elem_classes="results-section"):
                gr.HTML(
                    '<h3 style="color: #667eea; font-weight: bold; border-bottom: 2px solid #667eea; padding-bottom: 5px; margin-bottom: 20px;">📋 Application Ports & Manual Launch</h3>'
                )

                gr.HTML(
                    """
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                        <div style="color: #10b981; font-weight: 600; margin-bottom: 15px; font-size: 1.1rem;">🌐 Application URLs</div>
                        <div style="display: grid; gap: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(16, 185, 129, 0.2);">
                                <span style="color: #e2e8f0; font-weight: 500;">Central Hub:</span>
                                <code style="background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px; color: #a7f3d0;">localhost:7859</code>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(16, 185, 129, 0.2);">
                                <span style="color: #e2e8f0; font-weight: 500;">Face Detection:</span>
                                <code style="background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px; color: #a7f3d0;">localhost:7860</code>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(16, 185, 129, 0.2);">
                                <span style="color: #e2e8f0; font-weight: 500;">Face Reenactment:</span>
                                <code style="background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px; color: #a7f3d0;">localhost:7861</code>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0;">
                                <span style="color: #e2e8f0; font-weight: 500;">Deepfake Detection:</span>
                                <code style="background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px; color: #a7f3d0;">localhost:7862</code>
                            </div>
                        </div>
                    </div>
                    
                    <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 10px; padding: 20px;">
                        <div style="color: #f59e0b; font-weight: 600; margin-bottom: 15px; font-size: 1.1rem;">🔧 Manual Launch Commands</div>
                        <div style="color: #fcd34d; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.9rem; line-height: 1.6;">
                            <div style="margin-bottom: 8px;"><strong>cd app</strong></div>
                            <div style="margin-bottom: 8px;"><strong>python face_detection_app.py</strong> &nbsp;&nbsp;# Port 7860</div>
                            <div style="margin-bottom: 8px;"><strong>python reenactment_app.py</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Port 7861</div>
                            <div><strong>python deepfake_detection_app.py</strong> # Port 7862</div>
                        </div>
                    </div>
                    """
                )

        # Button click handlers
        face_detection_btn.click(
            fn=lambda: launch_app("Face Detection", 7860), outputs=status_display
        )

        reenactment_btn.click(
            fn=lambda: launch_app("Face Reenactment", 7861), outputs=status_display
        )

        deepfake_btn.click(
            fn=lambda: launch_app("Deepfake Detection", 7862), outputs=status_display
        )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7859,  # Central hub on port 7859
        share=False,
        show_api=False,
    )
