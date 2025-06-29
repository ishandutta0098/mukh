"""
Configuration loader for Mukh Apps design system.

Loads design configuration from YAML and provides methods to generate
CSS styles, theme configurations, and other design-related properties.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


class DesignConfigLoader:
    """Loads and manages design configuration for Mukh Apps."""

    def __init__(self, config_path: str = "app/config.yml"):
        """Initialize the design config loader.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Dictionary containing the configuration
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails.

        Returns:
            Default configuration dictionary
        """
        return {
            "theme": {
                "primary_hue": "violet",
                "secondary_hue": "blue",
                "neutral_hue": "slate",
            },
            "typography": {"font_family": "Inter"},
            "styling": {},
        }

    def get_theme_config(self) -> Dict[str, str]:
        """Get theme configuration for Gradio.

        Returns:
            Dictionary with theme hue settings
        """
        theme = self.config.get("theme", {})
        return {
            "primary_hue": theme.get("primary_hue", "violet"),
            "secondary_hue": theme.get("secondary_hue", "blue"),
            "neutral_hue": theme.get("neutral_hue", "slate"),
        }

    def get_theme_colors(self) -> Dict[str, str]:
        """Get theme colors for Gradio.

        Returns:
            Dictionary with color settings
        """
        colors = self.config.get("theme", {}).get("colors", {})
        return {
            "body_background_fill_dark": colors.get(
                "body_background_fill_dark", "#0f0f23"
            ),
            "background_fill_primary_dark": colors.get(
                "background_fill_primary_dark", "#1a1a2e"
            ),
            "background_fill_secondary_dark": colors.get(
                "background_fill_secondary_dark", "#16213e"
            ),
            "border_color_primary_dark": colors.get(
                "border_color_primary_dark", "#2d3748"
            ),
        }

    def get_font_family(self) -> str:
        """Get the font family configuration.

        Returns:
            Font family string
        """
        return self.config.get("typography", {}).get("font_family", "Inter")

    def get_google_fonts_url(self) -> str:
        """Get Google Fonts URL.

        Returns:
            Google Fonts URL string
        """
        return self.config.get("typography", {}).get(
            "google_fonts_url",
            "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
        )

    def get_styling_config(self) -> Dict[str, Any]:
        """Get the complete styling configuration.

        Returns:
            Dictionary containing all styling settings
        """
        return self.config.get("styling", {})

    def get_css_styles(self) -> str:
        """Generate comprehensive CSS styles from configuration.

        Returns:
            CSS string with all styling applied
        """
        styling = self.get_styling_config()
        typography = self.config.get("typography", {})
        theme_colors = self.config.get("theme", {}).get("colors", {})

        # Base container styles
        container = styling.get("container", {})

        # Header styles
        header = styling.get("header", {})

        # Card styles
        card = styling.get("card", {})
        card_hover = styling.get("card_hover", {})

        # Section styles
        section = styling.get("section", {})

        # Button styles
        button = styling.get("button", {})
        button_hover = styling.get("button_hover", {})

        # Input styles
        input_style = styling.get("input", {})
        input_focus = styling.get("input_focus", {})

        # Gallery styles
        gallery = styling.get("gallery", {})

        # Status styles
        status = styling.get("status", {})

        # Animation
        animation = styling.get("animation", {})

        css = f"""
        /* Import Google Fonts */
        @import url('{self.get_google_fonts_url()}');
        
        /* Global Container Styles */
        .gradio-container {{
            max-width: {container.get('max_width', '100%')} !important;
            width: 100% !important;
            margin: 0 !important;
            padding: {container.get('padding', '20px')} !important;
            background: {container.get('background', '#0a0a0a')} !important;
            font-family: '{self.get_font_family()}', sans-serif !important;
        }}
        
        .main {{
            max-width: 100% !important;
            width: 100% !important;
        }}
        
        /* Header Section Styling */
        .header-section {{
            background: {header.get('background', 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)')} !important;
            color: white !important;
            padding: {header.get('padding', '40px')} !important;
            border-radius: {header.get('border_radius', '15px')} !important;
            margin-bottom: {header.get('margin_bottom', '30px')} !important;
            text-align: {header.get('text_align', 'center')} !important;
            box-shadow: {header.get('box_shadow', '0 8px 32px rgba(102, 126, 234, 0.3)')} !important;
            transition: all {animation.get('duration_normal', '0.3s')} {animation.get('easing', 'ease')} !important;
        }}
        
        .header-section h1 {{
            font-size: {typography.get('header_lg', '2.5rem')} !important;
            font-weight: 700 !important;
            margin-bottom: 15px !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
            line-height: {typography.get('line_heights', {}).get('tight', '1.25')} !important;
        }}
        
        .header-section p {{
            font-size: {typography.get('body_lg', '1.2rem')} !important;
            margin-bottom: 10px !important;
            opacity: 0.95 !important;
            line-height: {typography.get('line_heights', {}).get('relaxed', '1.6')} !important;
        }}
        
        /* Main Header Styling */
        .main-header {{
            text-align: center !important;
            color: #f8fafc !important;
            font-size: {typography.get('header_xl', '3rem')} !important;
            font-weight: 800 !important;
            margin-bottom: 2rem !important;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.4) !important;
        }}
        
        /* Card Styling */
        .app-card {{
            background: {card.get('background', 'linear-gradient(145deg, #1a1a2e, #16213e)')} !important;
            border-radius: {card.get('border_radius', '16px')} !important;
            padding: {card.get('padding', '24px')} !important;
            margin: {card.get('margin', '15px 0')} !important;
            border: {card.get('border', '1px solid #2d3748')} !important;
            box-shadow: {card.get('box_shadow', '0 4px 20px rgba(0, 0, 0, 0.4)')} !important;
            transition: {card.get('transition', 'all 0.3s ease')} !important;
        }}
        
        .app-card:hover {{
            transform: {card_hover.get('transform', 'translateY(-4px)')} !important;
            box-shadow: {card_hover.get('box_shadow', '0 12px 40px rgba(102, 126, 234, 0.2)')} !important;
            border-color: {card_hover.get('border_color', '#667eea')} !important;
        }}
        
        /* Section Styling */
        .search-section,
        .results-section,
        .gallery-section {{
            background: {section.get('background', '#1a1a2e')} !important;
            padding: {section.get('padding', '25px')} !important;
            border-radius: {section.get('border_radius', '12px')} !important;
            margin-bottom: {section.get('margin_bottom', '20px')} !important;
            border: {section.get('border', '1px solid #2d3748')} !important;
            box-shadow: {section.get('box_shadow', '0 2px 15px rgba(0, 0, 0, 0.3)')} !important;
            height: fit-content !important;
        }}
        
        /* Typography Styles */
        .app-title {{
            font-size: {typography.get('header_sm', '1.5rem')} !important;
            font-weight: 600 !important;
            color: #f8fafc !important;
            margin-bottom: 12px !important;
            line-height: {typography.get('line_heights', {}).get('tight', '1.25')} !important;
        }}
        
        .app-description {{
            color: #cbd5e1 !important;
            margin-bottom: 16px !important;
            line-height: {typography.get('line_heights', {}).get('relaxed', '1.6')} !important;
            font-size: {typography.get('body_md', '1rem')} !important;
        }}
        
        .app-features {{
            color: #94a3b8 !important;
            font-size: {typography.get('body_sm', '0.9rem')} !important;
            margin-bottom: 20px !important;
            line-height: {typography.get('line_heights', {}).get('normal', '1.5')} !important;
        }}
        
        .section-header {{
            color: {theme_colors.get('accent_color', '#667eea')} !important;
            font-weight: bold !important;
            border-bottom: 2px solid {theme_colors.get('accent_color', '#667eea')} !important;
            padding-bottom: 5px !important;
            margin-bottom: 15px !important;
            font-size: {typography.get('header_sm', '1.5rem')} !important;
        }}
        
        /* Button Styling */
        .primary-button,
        button[variant="primary"] {{
            background: {button.get('primary_gradient', 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)')} !important;
            border: none !important;
            border-radius: {button.get('border_radius', '8px')} !important;
            padding: {button.get('padding', '12px 24px')} !important;
            font-weight: {button.get('font_weight', '600')} !important;
            font-size: {typography.get('body_md', '1rem')} !important;
            color: white !important;
            box-shadow: {button.get('box_shadow', '0 4px 15px rgba(102, 126, 234, 0.4)')} !important;
            transition: {button.get('transition', 'all 0.3s ease')} !important;
            cursor: pointer !important;
        }}
        
        .primary-button:hover,
        button[variant="primary"]:hover {{
            transform: {button_hover.get('transform', 'translateY(-2px)')} !important;
            box-shadow: {button_hover.get('box_shadow', '0 6px 25px rgba(102, 126, 234, 0.6)')} !important;
        }}
        
        /* Input and Textarea Styling */
        .gradio-textbox textarea,
        .gradio-textbox input {{
            background: {input_style.get('background', '#16213e')} !important;
            border: {input_style.get('border', '2px solid #2d3748')} !important;
            border-radius: {input_style.get('border_radius', '8px')} !important;
            color: {input_style.get('color', '#e2e8f0')} !important;
            font-family: '{self.get_font_family()}', sans-serif !important;
            line-height: {typography.get('line_heights', {}).get('normal', '1.5')} !important;
        }}
        
        .gradio-textbox textarea:focus,
        .gradio-textbox input:focus {{
            border-color: {input_focus.get('border_color', '#667eea')} !important;
            background: {input_focus.get('background', '#1a1a2e')} !important;
            box-shadow: {input_focus.get('box_shadow', '0 0 0 3px rgba(102, 126, 234, 0.1)')} !important;
            outline: none !important;
        }}
        
        /* Gallery Styling */
        .gallery-container {{
            margin-top: 15px !important;
            width: 100% !important;
            background: {gallery.get('background', '#1a1a2e')} !important;
            border-radius: {gallery.get('border_radius', '12px')} !important;
            padding: {gallery.get('padding', '20px')} !important;
            border: {gallery.get('border', '1px solid #2d3748')} !important;
        }}
        
        .gallery img {{
            border-radius: 8px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
            transition: all {animation.get('duration_normal', '0.3s')} {animation.get('easing', 'ease')} !important;
        }}
        
        .gallery img:hover {{
            transform: scale(1.02) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
        }}
        
        /* Status Message Styling */
        .status-message {{
            margin-top: 20px !important;
            padding: 12px !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-family: '{self.get_font_family()}', sans-serif !important;
        }}
        
        .status-success {{
            background: {status.get('success_bg', '#064e3b')} !important;
            border: 1px solid {status.get('success_border', '#10b981')} !important;
            color: #a7f3d0 !important;
        }}
        
        .status-warning {{
            background: {status.get('warning_bg', '#451a03')} !important;
            border: 1px solid {status.get('warning_border', '#f59e0b')} !important;
            color: #fcd34d !important;
        }}
        
        .status-error {{
            background: {status.get('error_bg', '#450a0a')} !important;
            border: 1px solid {status.get('error_border', '#ef4444')} !important;
            color: #fca5a5 !important;
        }}
        
        /* Output Text Styling */
        .output-text {{
            font-family: '{self.get_font_family()}', sans-serif !important;
            line-height: {typography.get('line_heights', {}).get('relaxed', '1.6')} !important;
            color: #e2e8f0 !important;
            background: #16213e !important;
            border: 1px solid #2d3748 !important;
            border-radius: 8px !important;
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .gradio-container {{
                padding: 10px !important;
            }}
            
            .header-section {{
                padding: 20px !important;
            }}
            
            .header-section h1 {{
                font-size: {typography.get('header_md', '2rem')} !important;
            }}
            
            .app-card {{
                padding: 16px !important;
                margin: 10px 0 !important;
            }}
            
            .search-section,
            .results-section,
            .gallery-section {{
                padding: 15px !important;
            }}
        }}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: #1a1a2e;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: #667eea;
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: #764ba2;
        }}
        """

        return css


# Global instance
design_config = DesignConfigLoader()
