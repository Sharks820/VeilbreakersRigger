#!/usr/bin/env python3
"""VEILBREAKERS RIGGER - UNIFIED UI v4.0

Combines all features from:
- veilbreakers_rigger_ui.py (Professional UI with undo/redo, tabs, exports)
- working_ui.py (Florence-2 Smart Detect, Segment Browser, Box Selection)
- simple_rigger_ui.py (Graceful degradation)
"""

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import json
import tempfile
import shutil
import os
from typing import Optional, List, Tuple, Dict, Any

# =============================================================================
# GRACEFUL DEGRADATION
# =============================================================================
RIGGER_AVAILABLE = True
try:
    from veilbreakers_rigger import (
        VeilbreakersRigger,
        BODY_TEMPLATES,
        InpaintQuality,
        ExportFormat,
        BodyPart,
        Point
    )
except Exception as e:
    RIGGER_AVAILABLE = False
    print(f"WARNING: Could not import VeilbreakersRigger: {e}")

# =============================================================================
# GLOBAL STATE
# =============================================================================

class AppState:
    """Application state manager with full undo/redo and segment browsing"""

    def __init__(self):
        self.rigger: Optional[VeilbreakersRigger] = None
        self.mode = "select"
        self.current_part_name = ""
        self.history: List[np.ndarray] = []
        self.history_index = -1
        self.models_loaded = False

        # Segment browser state
        self.segment_count = 0
        self.current_segment_index = 0

        # Box selection state
        self.box_first_click: Optional[Tuple[int, int]] = None

    def init_rigger(self, sam_size: str = "large"):
        """Initialize or reinitialize the rigger"""
        if not RIGGER_AVAILABLE:
            print("ERROR: VeilbreakersRigger not available")
            return False

        try:
            self.rigger = VeilbreakersRigger(
                output_dir="./output",
                sam_size=sam_size,
                use_fallback=True
            )
            return True
        except Exception as e:
            print(f"Error initializing rigger: {e}")
            try:
                self.rigger = VeilbreakersRigger(
                    output_dir="./output",
                    sam_size="tiny",
                    use_fallback=True
                )
                return True
            except:
                return False

    def preload_models(self):
        """Pre-load AI models at startup for faster first use"""
        if self.models_loaded or self.rigger is None:
            return

        print("Pre-loading AI models (this takes 1-2 minutes on CPU)...")
        try:
            self.rigger.segmenter.load()
            self.models_loaded = True
            print("AI models loaded successfully!")
        except Exception as e:
            print(f"Warning: Model pre-load failed: {e}")

    def save_state(self):
        """Save current state to history"""
        if self.rigger and self.rigger.current_rig:
            self.history = self.history[:self.history_index + 1]
            self.history.append(self.rigger.get_working_image().copy())
            self.history_index = len(self.history) - 1
            if len(self.history) > 20:
                self.history = self.history[-20:]
                self.history_index = len(self.history) - 1

    def undo(self) -> Optional[np.ndarray]:
        if self.history_index > 0:
            self.history_index -= 1
            return self.history[self.history_index]
        return None

    def redo(self) -> Optional[np.ndarray]:
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            return self.history[self.history_index]
        return None

    def reset(self):
        if self.rigger:
            self.rigger.reset()
        self.mode = "select"
        self.current_part_name = ""
        self.history = []
        self.history_index = -1
        self.segment_count = 0
        self.current_segment_index = 0
        self.box_first_click = None

STATE = AppState()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_visualization(show_mask: bool = True, mask_color: tuple = (255, 100, 100)) -> Optional[np.ndarray]:
    """Create visualization of current state"""
    if STATE.rigger is None or STATE.rigger.current_rig is None:
        return None

    vis = STATE.rigger.get_working_image().copy()

    if show_mask and STATE.rigger.current_mask is not None:
        mask = STATE.rigger.current_mask
        overlay = np.zeros_like(vis)
        overlay[mask > 0] = mask_color
        vis = (vis * 0.7 + overlay * 0.3).astype(np.uint8)

    return vis

def get_parts_table() -> List[List[str]]:
    """Get parts as table data"""
    if STATE.rigger is None:
        return []
    parts = STATE.rigger.get_parts()
    return [[p.name, str(p.z_index), p.parent or "None"] for p in parts]

def get_part_choices() -> List[str]:
    """Get part names for dropdown"""
    if STATE.rigger is None:
        return [""]
    return [""] + [p.name for p in STATE.rigger.get_parts()]

def get_parts_markdown() -> str:
    """Get parts as markdown list"""
    if STATE.rigger is None or STATE.rigger.current_rig is None:
        return "No parts saved yet. Load an image and detect parts first."

    parts = STATE.rigger.get_parts()
    if not parts:
        return "No parts saved yet."

    lines = [f"**{len(parts)} parts saved:**"]
    for i, p in enumerate(parts):
        lines.append(f"{i+1}. **{p.name}** (z={p.z_index})")

    return "\n".join(lines)

# =============================================================================
# MAIN UI FUNCTIONS
# =============================================================================

def load_image(image, sam_size: str):
    """Load an image into the rigger"""
    if image is None:
        return None, "No image provided", [], gr.update(choices=[""]), None

    if STATE.rigger is None:
        if not STATE.init_rigger(sam_size):
            return None, "Failed to initialize rigger", [], gr.update(choices=[""]), None

    STATE.reset()

    try:
        if isinstance(image, str):
            STATE.rigger.load_image(image)
        else:
            STATE.rigger.load_image_array(image, "monster")

        # Set image in segmenter for click-to-segment
        STATE.rigger.segmenter.set_image(STATE.rigger.current_rig.original_image)

        STATE.save_state()

        vis = create_visualization(show_mask=False)
        parts = get_parts_table()
        choices = get_part_choices()

        return (
            vis,
            f"Loaded image ({STATE.rigger.current_rig.original_image.shape[1]}x{STATE.rigger.current_rig.original_image.shape[0]}). Use Smart Detect or click to select parts.",
            parts,
            gr.update(choices=choices),
            STATE.rigger.get_original_image()
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}", [], gr.update(choices=[""]), None


def on_image_click(image, evt: gr.SelectData, mode: str, box_mode: bool):
    """Handle click on image - supports regular click and box mode"""
    if STATE.rigger is None or STATE.rigger.current_rig is None:
        return None, "Load an image first!"

    x, y = evt.index

    # Handle box selection mode
    if box_mode:
        if STATE.box_first_click is None:
            STATE.box_first_click = (x, y)
            return image, f"Box start: ({x}, {y}). Click again for end point."
        else:
            x1, y1 = STATE.box_first_click
            x2, y2 = x, y
            STATE.box_first_click = None

            try:
                STATE.rigger.box_segment(x1, y1, x2, y2)
                vis = create_visualization(mask_color=(255, 50, 50))
                return vis, f"Box segment: ({x1},{y1}) to ({x2},{y2}). Name it and click 'Add Part'."
            except Exception as e:
                return image, f"Box segment error: {str(e)}"

    # Regular click mode
    try:
        if mode == "select" or STATE.rigger.current_mask is None:
            # Make sure segmenter has the working image
            working = STATE.rigger.get_working_image()
            if working is not None:
                STATE.rigger.segmenter.set_image(working)
            STATE.rigger.click_segment(x, y)
            status = f"Selected at ({x}, {y}) - Click 'Add Part' or refine"
        elif mode == "add":
            STATE.rigger.refine_add(x, y)
            status = f"Added point at ({x}, {y})"
        elif mode == "subtract":
            STATE.rigger.refine_subtract(x, y)
            status = f"Removed point at ({x}, {y})"
        else:
            STATE.rigger.click_segment(x, y)
            status = f"Selected at ({x}, {y})"

        vis = create_visualization()
        return vis, status

    except Exception as e:
        return image, f"Error: {str(e)}"


# =============================================================================
# SMART DETECTION (Florence-2)
# =============================================================================

def smart_detect_parts(image, prompt: str, threshold: float):
    """
    BEST detection - uses Florence-2 unified vision model.
    No prompt required! Florence-2 finds AND locates all parts automatically.
    """
    if image is None:
        return None, "Upload and load an image first", [], gr.update(choices=[""])

    if STATE.rigger is None or STATE.rigger.current_rig is None:
        return image, "Click 'Load Image' first", [], gr.update(choices=[""])

    if not STATE.models_loaded:
        return image, "AI models still loading... wait a moment and try again", [], gr.update(choices=[""])

    quality_map = {
        "Fast (OpenCV)": InpaintQuality.FAST,
        "Standard (LaMa)": InpaintQuality.STANDARD,
        "High (LaMa x2)": InpaintQuality.HIGH,
        "Ultra (Stable Diffusion)": InpaintQuality.ULTRA
    }

    try:
        print(f"Smart-detecting with Florence-2 (threshold={threshold})")

        text_prompt = prompt.strip() if prompt and prompt.strip() else None
        parts = STATE.rigger.smart_detect(
            text_prompt=text_prompt,
            use_florence=True,
            box_threshold=threshold,
            inpaint_quality=quality_map.get("Standard (LaMa)", InpaintQuality.STANDARD)
        )

        STATE.save_state()

        if len(parts) == 0:
            return (
                create_visualization(show_mask=False),
                "No parts detected. Try 'Find All Segments' or click directly on the image.",
                get_parts_table(),
                gr.update(choices=get_part_choices())
            )

        return (
            create_visualization(show_mask=False),
            f"Detected {len(parts)} parts: {', '.join(p.name for p in parts)}. Click 'Export' when ready.",
            get_parts_table(),
            gr.update(choices=get_part_choices())
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return create_visualization(show_mask=False), f"Detection error: {str(e)}", get_parts_table(), gr.update(choices=get_part_choices())


def auto_detect(prompt: str, box_thresh: float, text_thresh: float, quality: str):
    """Auto-detect parts from text prompt (Grounding DINO)"""
    if STATE.rigger is None or STATE.rigger.current_rig is None:
        return None, "Load an image first!", [], gr.update(choices=[""])

    if not prompt:
        return None, "Enter a detection prompt!", [], gr.update(choices=[""])

    quality_map = {
        "Fast (OpenCV)": InpaintQuality.FAST,
        "Standard (LaMa)": InpaintQuality.STANDARD,
        "High (LaMa x2)": InpaintQuality.HIGH,
        "Ultra (Stable Diffusion)": InpaintQuality.ULTRA
    }

    try:
        parts = STATE.rigger.auto_detect(
            prompt,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            inpaint_quality=quality_map.get(quality, InpaintQuality.STANDARD)
        )

        STATE.save_state()

        if not parts:
            return (
                create_visualization(show_mask=False),
                "No parts detected - try different prompt or thresholds",
                get_parts_table(),
                gr.update(choices=get_part_choices())
            )

        return (
            create_visualization(show_mask=False),
            f"Detected {len(parts)} parts: {', '.join(p.name for p in parts)}",
            get_parts_table(),
            gr.update(choices=get_part_choices())
        )
    except Exception as e:
        return create_visualization(show_mask=False), f"Error: {str(e)}", get_parts_table(), gr.update(choices=get_part_choices())


def redetect_single_part(part_prompt: str, threshold: float):
    """Re-detect a single part with custom settings"""
    if STATE.rigger is None or STATE.rigger.current_rig is None:
        return None, "Load an image first"

    if not STATE.models_loaded:
        return None, "AI models still loading..."

    if not part_prompt or not part_prompt.strip():
        return None, "Enter a single part name like 'head' or 'arm'"

    try:
        print(f"Re-detecting: {part_prompt} (threshold={threshold})")

        # Set segmenter to original image for fresh detection
        STATE.rigger.segmenter.set_image(STATE.rigger.current_rig.original_image)

        detections = STATE.rigger.segmenter.auto_detect(
            part_prompt.strip(),
            box_threshold=threshold,
            text_threshold=threshold
        )

        if not detections:
            return None, f"'{part_prompt}' not detected. Try lower threshold or different term."

        # Take the best detection
        name, mask, confidence = detections[0]
        print(f"Found {name} with confidence {confidence:.2f}")

        # Show mask overlay on original
        vis = STATE.rigger.current_rig.original_image.copy()
        overlay = np.zeros_like(vis)
        overlay[mask > 0] = [50, 255, 50]  # Green overlay
        vis = (vis.astype(float) * 0.6 + overlay.astype(float) * 0.4).astype(np.uint8)

        # Store mask for adding
        STATE.rigger.current_mask = mask

        return vis, f"Found '{name}' (confidence: {confidence:.1%}). Enter name and click 'Add Part' to save."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Re-detect error: {str(e)}"


# =============================================================================
# SEGMENT BROWSER
# =============================================================================

def segment_everything():
    """Find all segments in the image automatically"""
    if STATE.rigger is None or STATE.rigger.current_rig is None:
        return None, "Load an image first", 0, 0

    try:
        print("Finding all segments...")
        segments = STATE.rigger.segment_everything(min_mask_area=500)
        STATE.segment_count = len(segments)
        STATE.current_segment_index = 0

        if STATE.segment_count == 0:
            return None, "No segments found. Try clicking directly on the image.", 0, 0

        # Show first segment
        vis = STATE.rigger.get_segment_preview(0)
        STATE.rigger.select_segment(0)

        return vis, f"Found {STATE.segment_count} segments! Use Prev/Next to browse, then 'Add Part' to save.", STATE.segment_count, 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Segmentation error: {str(e)}", 0, 0


def show_segment(index: int):
    """Show a specific segment by index"""
    if STATE.rigger is None or STATE.rigger.current_rig is None:
        return None, "No image loaded", 0

    if STATE.segment_count == 0:
        return None, "Run 'Find All Segments' first", 0

    # Clamp index
    index = max(0, min(int(index), STATE.segment_count - 1))
    STATE.current_segment_index = index

    try:
        vis = STATE.rigger.get_segment_preview(index)
        STATE.rigger.select_segment(index)
        mask = STATE.rigger.current_mask
        area = (mask > 0).sum() if mask is not None else 0
        return vis, f"Segment {index + 1}/{STATE.segment_count} (area: {area:,} px). Name it and click 'Add Part'.", index
    except Exception as e:
        return None, f"Error: {str(e)}", index


def prev_segment():
    """Show previous segment"""
    new_index = max(0, STATE.current_segment_index - 1)
    return show_segment(new_index)


def next_segment():
    """Show next segment"""
    new_index = min(STATE.segment_count - 1, STATE.current_segment_index + 1)
    return show_segment(new_index)


# =============================================================================
# PART MANAGEMENT
# =============================================================================

def add_part(name: str, z_index: int, parent: str, pivot: str, quality: str):
    """Add current selection as a part"""
    if STATE.rigger is None or STATE.rigger.current_mask is None:
        return None, "Select a region first!", [], gr.update(choices=[""])

    if not name:
        return None, "Enter a part name!", [], gr.update(choices=[""])

    quality_map = {
        "Fast (OpenCV)": InpaintQuality.FAST,
        "Standard (LaMa)": InpaintQuality.STANDARD,
        "High (LaMa x2)": InpaintQuality.HIGH,
        "Ultra (Stable Diffusion)": InpaintQuality.ULTRA
    }

    pivot_map = {
        "Center": "center", "Top Center": "top_center", "Bottom Center": "bottom_center",
        "Left Center": "left_center", "Right Center": "right_center",
        "Top Left": "top_left", "Top Right": "top_right",
        "Bottom Left": "bottom_left", "Bottom Right": "bottom_right"
    }

    try:
        STATE.rigger.add_part(
            name=name,
            z_index=z_index,
            parent=parent if parent else None,
            pivot=pivot_map.get(pivot, "center"),
            inpaint_quality=quality_map.get(quality, InpaintQuality.STANDARD)
        )

        STATE.save_state()

        # Update segmenter with new working image
        vis = STATE.rigger.get_working_image()
        if vis is not None:
            STATE.rigger.segmenter.set_image(vis)

        return (
            create_visualization(show_mask=False),
            f"Added part: {name}",
            get_parts_table(),
            gr.update(choices=get_part_choices())
        )
    except Exception as e:
        return create_visualization(), f"Error: {str(e)}", get_parts_table(), gr.update(choices=get_part_choices())


def clear_selection():
    """Clear current selection"""
    if STATE.rigger:
        STATE.rigger.clear_selection()
    STATE.box_first_click = None
    return create_visualization(show_mask=False), "Selection cleared"


def reset_all():
    """Reset everything"""
    STATE.reset()
    return None, "Reset complete", [], gr.update(choices=[""]), None


def undo_action():
    """Undo last action"""
    result = STATE.undo()
    if result is not None:
        return result, "Undo"
    return None, "Nothing to undo"


def redo_action():
    """Redo last action"""
    result = STATE.redo()
    if result is not None:
        return result, "Redo"
    return None, "Nothing to redo"


def get_preset_info(preset_name: str) -> str:
    """Get info about a preset"""
    if not RIGGER_AVAILABLE:
        return "Rigger not available"
    if preset_name in BODY_TEMPLATES:
        template = BODY_TEMPLATES[preset_name]
        parts = template.get("parts", [])
        return f"**{preset_name.title()}**\n\nParts: {', '.join(parts)}"
    return ""


def apply_preset(preset_name: str, quality: str):
    """Apply a body preset"""
    if STATE.rigger is None or STATE.rigger.current_rig is None:
        return None, "Load an image first!", [], gr.update(choices=[""])

    quality_map = {
        "Fast (OpenCV)": InpaintQuality.FAST,
        "Standard (LaMa)": InpaintQuality.STANDARD,
        "High (LaMa x2)": InpaintQuality.HIGH,
        "Ultra (Stable Diffusion)": InpaintQuality.ULTRA
    }

    try:
        parts = STATE.rigger.apply_template(
            preset_name,
            inpaint_quality=quality_map.get(quality, InpaintQuality.STANDARD)
        )

        STATE.save_state()

        return (
            create_visualization(show_mask=False),
            f"Applied {preset_name} preset - {len(parts)} parts",
            get_parts_table(),
            gr.update(choices=get_part_choices())
        )
    except Exception as e:
        return create_visualization(show_mask=False), f"Error: {str(e)}", get_parts_table(), gr.update(choices=get_part_choices())


def update_part(name: str, z_index: int, parent: str, pivot: str):
    """Update a part's properties"""
    if STATE.rigger is None or not name:
        return None, "Select a part to edit", []

    pivot_map = {
        "Center": "center", "Top Center": "top_center", "Bottom Center": "bottom_center",
        "Left Center": "left_center", "Right Center": "right_center",
        "Top Left": "top_left", "Top Right": "top_right",
        "Bottom Left": "bottom_left", "Bottom Right": "bottom_right"
    }

    try:
        STATE.rigger.update_part(
            name=name,
            z_index=z_index,
            parent=parent if parent else None,
            pivot=pivot_map.get(pivot, "center")
        )

        return (
            create_visualization(show_mask=False),
            f"Updated part: {name}",
            get_parts_table()
        )
    except Exception as e:
        return create_visualization(show_mask=False), f"Error: {str(e)}", get_parts_table()


def remove_part(name: str):
    """Remove a part"""
    if STATE.rigger is None or not name:
        return None, "Select a part to remove", [], gr.update(choices=[""])

    try:
        STATE.rigger.remove_part(name)
        STATE.save_state()

        return (
            create_visualization(show_mask=False),
            f"Removed part: {name}",
            get_parts_table(),
            gr.update(choices=get_part_choices())
        )
    except Exception as e:
        return create_visualization(show_mask=False), f"Error: {str(e)}", get_parts_table(), gr.update(choices=get_part_choices())


def export_rig(monster_name: str, format_choice: str):
    """Export the rig"""
    if STATE.rigger is None or not STATE.rigger.get_parts():
        return "No parts to export!", None

    if not monster_name:
        monster_name = "monster"

    format_map = {
        "Godot Scene (.tscn)": ExportFormat.GODOT,
        "Spine JSON": ExportFormat.SPINE,
        "PNG Layers": ExportFormat.PNG_LAYERS
    }

    try:
        output_path = STATE.rigger.export(
            name=monster_name,
            format=format_map.get(format_choice, ExportFormat.GODOT)
        )

        return f"Exported to: {output_path}", output_path
    except Exception as e:
        return f"Export error: {str(e)}", None


# =============================================================================
# CREATE UI
# =============================================================================

def create_ui():
    """Create the Gradio interface"""

    if not RIGGER_AVAILABLE:
        with gr.Blocks(title="VEILBREAKERS Monster Rigger - ERROR") as app:
            gr.Markdown("# VEILBREAKERS Monster Rigger")
            gr.Markdown("## ⚠️ ERROR: VeilbreakersRigger could not be loaded")
            gr.Markdown("Check the console for error details. Make sure all dependencies are installed.")
        return app

    with gr.Blocks(
        title="VEILBREAKERS Monster Rigger",
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate")
    ) as app:

        gr.Markdown("# VEILBREAKERS Monster Rigger v4.0")
        gr.Markdown("*AI-powered monster segmentation with Florence-2 Smart Detection*")

        with gr.Row():
            # LEFT PANEL
            with gr.Column(scale=2):

                with gr.Tab("Workspace"):
                    main_image = gr.Image(
                        label="Monster Image (Upload or Click to Select)",
                        type="numpy",
                        height=550,
                        interactive=False,
                        sources=["upload", "clipboard"]
                    )

                    with gr.Row():
                        load_btn = gr.Button("Load Image", variant="primary", size="sm")
                        clear_btn = gr.Button("Clear Selection", size="sm")
                        reset_btn = gr.Button("Reset All", variant="stop", size="sm")

                    with gr.Row():
                        undo_btn = gr.Button("Undo", size="sm")
                        redo_btn = gr.Button("Redo", size="sm")

                    status_text = gr.Textbox(label="Status", interactive=False, lines=2)

                with gr.Tab("Selection Mode"):
                    gr.Markdown("### Click Mode\n- **Select**: Start a new selection\n- **Add (+)**: Add to current selection\n- **Subtract (-)**: Remove from current selection")

                    mode_radio = gr.Radio(
                        choices=["select", "add", "subtract"],
                        value="select",
                        label="Click Mode",
                        interactive=True
                    )

                    gr.Markdown("---")
                    gr.Markdown("### Box Selection")
                    box_mode = gr.Checkbox(
                        label="Box Selection Mode (2-click)",
                        value=False,
                        info="Click two corners to draw a bounding box"
                    )

                with gr.Tab("Settings"):
                    sam_size = gr.Dropdown(
                        choices=["tiny", "small", "base", "large"],
                        value="large",
                        label="SAM Model Size",
                        info="Larger = more accurate, slower"
                    )

                    inpaint_quality = gr.Dropdown(
                        choices=["Fast (OpenCV)", "Standard (LaMa)", "High (LaMa x2)", "Ultra (Stable Diffusion)"],
                        value="Standard (LaMa)",
                        label="Inpainting Quality"
                    )

            # RIGHT PANEL
            with gr.Column(scale=1):

                with gr.Tab("Smart Detect"):
                    gr.Markdown("### 🤖 Florence-2 (Best)")
                    gr.Markdown("*No prompt needed - AI finds all body parts automatically*")

                    smart_threshold = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="Detection Threshold")
                    smart_detect_btn = gr.Button("Smart Detect (Florence-2)", variant="primary", size="lg")

                    gr.Markdown("---")
                    gr.Markdown("### 🔍 Segment Browser")
                    gr.Markdown("*Find ALL segments and browse through them*")

                    segment_all_btn = gr.Button("Find All Segments", variant="secondary")

                    with gr.Row():
                        segment_count = gr.Number(label="Found", value=0, interactive=False, scale=1)
                        seg_index = gr.Number(label="Current #", value=0, minimum=0, step=1, scale=1)

                    with gr.Row():
                        prev_seg_btn = gr.Button("◀ Prev", size="sm")
                        next_seg_btn = gr.Button("Next ▶", size="sm")

                    gr.Markdown("---")
                    gr.Markdown("### 🔎 Re-detect Single Part")
                    single_part = gr.Textbox(label="Part Name", placeholder="e.g., head")
                    single_threshold = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="Threshold")
                    redetect_btn = gr.Button("Detect Single Part")

                with gr.Tab("Text Detect"):
                    gr.Markdown("### Text-Based Detection")
                    gr.Markdown("*Use if Smart Detection misses something*")

                    preset_dropdown = gr.Dropdown(
                        choices=list(BODY_TEMPLATES.keys()) if RIGGER_AVAILABLE else [],
                        value="quadruped" if RIGGER_AVAILABLE else None,
                        label="Body Template"
                    )

                    preset_info = gr.Markdown()
                    apply_preset_btn = gr.Button("Apply Preset", variant="primary")

                    gr.Markdown("---\n### Custom Prompt")

                    custom_prompt = gr.Textbox(
                        label="Detection Prompt",
                        placeholder="head . body . arms . legs . tail",
                        info="Separate parts with ' . '"
                    )

                    with gr.Row():
                        box_thresh = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Box Threshold")
                        text_thresh = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Text Threshold")

                    detect_btn = gr.Button("Detect Parts", variant="primary")

                with gr.Tab("Add Part"):
                    gr.Markdown("### Add Selection as Part")
                    gr.Markdown("Click on the image to select, then fill in details below.")

                    part_name = gr.Textbox(label="Part Name", placeholder="e.g., head, arm_left, tail")
                    z_index = gr.Slider(0, 20, value=0, step=1, label="Z-Index (Layer Order)", info="Higher = in front")
                    parent_dropdown = gr.Dropdown(choices=[""], value="", label="Parent Part", allow_custom_value=True)

                    pivot_type = gr.Dropdown(
                        choices=["Center", "Top Center", "Bottom Center", "Left Center", "Right Center",
                                 "Top Left", "Top Right", "Bottom Left", "Bottom Right"],
                        value="Center",
                        label="Pivot Point"
                    )

                    add_part_btn = gr.Button("Add Part", variant="primary")

                with gr.Tab("Edit Parts"):
                    gr.Markdown("### Edit Existing Parts")

                    edit_part_dropdown = gr.Dropdown(choices=[""], label="Select Part")
                    part_preview = gr.Image(label="Part Preview", height=150)

                    edit_z = gr.Slider(0, 20, value=0, step=1, label="Z-Index")
                    edit_parent = gr.Dropdown(choices=[""], label="Parent", allow_custom_value=True)
                    edit_pivot = gr.Dropdown(
                        choices=["Center", "Top Center", "Bottom Center", "Left Center", "Right Center",
                                 "Top Left", "Top Right", "Bottom Left", "Bottom Right"],
                        value="Center",
                        label="Pivot"
                    )

                    with gr.Row():
                        update_btn = gr.Button("Update", variant="primary")
                        remove_btn = gr.Button("Remove", variant="stop")

                with gr.Tab("Export"):
                    gr.Markdown("### Export Rig")

                    monster_name = gr.Textbox(label="Monster Name", placeholder="my_monster")
                    export_format = gr.Dropdown(
                        choices=["Godot Scene (.tscn)", "Spine JSON", "PNG Layers"],
                        value="Godot Scene (.tscn)",
                        label="Export Format"
                    )

                    export_btn = gr.Button("Export Rig", variant="primary")
                    export_status = gr.Textbox(label="Export Status", interactive=False)
                    download_file = gr.File(label="Download")

                with gr.Tab("Parts List"):
                    parts_table = gr.Dataframe(
                        headers=["Name", "Z-Index", "Parent"],
                        label="Current Parts",
                        interactive=False
                    )

                    parts_markdown = gr.Markdown("No parts saved yet.")
                    refresh_parts_btn = gr.Button("Refresh Parts List", size="sm")

        # Hidden state
        original_image = gr.State(None)

        # EVENT HANDLERS

        # Load/Reset
        load_btn.click(fn=load_image, inputs=[main_image, sam_size], outputs=[main_image, status_text, parts_table, parent_dropdown, original_image])
        clear_btn.click(fn=clear_selection, outputs=[main_image, status_text])
        reset_btn.click(fn=reset_all, outputs=[main_image, status_text, parts_table, parent_dropdown, original_image])
        undo_btn.click(fn=undo_action, outputs=[main_image, status_text])
        redo_btn.click(fn=redo_action, outputs=[main_image, status_text])

        # Image click - supports box mode
        main_image.select(fn=on_image_click, inputs=[main_image, mode_radio, box_mode], outputs=[main_image, status_text])

        # Smart Detection
        smart_detect_btn.click(fn=smart_detect_parts, inputs=[main_image, custom_prompt, smart_threshold], outputs=[main_image, status_text, parts_table, parent_dropdown])

        # Segment Browser
        segment_all_btn.click(fn=segment_everything, outputs=[main_image, status_text, segment_count, seg_index])
        prev_seg_btn.click(fn=prev_segment, outputs=[main_image, status_text, seg_index])
        next_seg_btn.click(fn=next_segment, outputs=[main_image, status_text, seg_index])
        seg_index.change(fn=show_segment, inputs=[seg_index], outputs=[main_image, status_text, seg_index])

        # Single Part Re-detect
        redetect_btn.click(fn=redetect_single_part, inputs=[single_part, single_threshold], outputs=[main_image, status_text])

        # Text Detection
        preset_dropdown.change(fn=get_preset_info, inputs=[preset_dropdown], outputs=[preset_info])
        apply_preset_btn.click(fn=apply_preset, inputs=[preset_dropdown, inpaint_quality], outputs=[main_image, status_text, parts_table, parent_dropdown])
        detect_btn.click(fn=auto_detect, inputs=[custom_prompt, box_thresh, text_thresh, inpaint_quality], outputs=[main_image, status_text, parts_table, parent_dropdown])

        # Part Management
        add_part_btn.click(fn=add_part, inputs=[part_name, z_index, parent_dropdown, pivot_type, inpaint_quality], outputs=[main_image, status_text, parts_table, parent_dropdown])

        def update_edit_fields(name):
            if STATE.rigger is None or not name:
                return 0, "", "Center", None
            part = STATE.rigger.get_part(name)
            if part:
                return part.z_index, part.parent, "Center", part.image
            return 0, "", "Center", None

        edit_part_dropdown.change(fn=update_edit_fields, inputs=[edit_part_dropdown], outputs=[edit_z, edit_parent, edit_pivot, part_preview])
        update_btn.click(fn=update_part, inputs=[edit_part_dropdown, edit_z, edit_parent, edit_pivot], outputs=[main_image, status_text, parts_table])
        remove_btn.click(fn=remove_part, inputs=[edit_part_dropdown], outputs=[main_image, status_text, parts_table, parent_dropdown])
        parts_table.change(fn=lambda: gr.update(choices=get_part_choices()), outputs=[edit_part_dropdown])

        # Parts List
        refresh_parts_btn.click(fn=get_parts_markdown, outputs=[parts_markdown])

        # Export
        export_btn.click(fn=export_rig, inputs=[monster_name, export_format], outputs=[export_status, download_file])

        # Load preset info on app load
        app.load(fn=lambda: get_preset_info("quadruped"), outputs=[preset_info])

    return app


def launch_ui():
    """Launch the UI"""
    print("=" * 70)
    print("           VEILBREAKERS MONSTER RIGGER v4.0 (UNIFIED)")
    print("=" * 70)

    if not RIGGER_AVAILABLE:
        print("\n⚠️  ERROR: VeilbreakersRigger could not be imported!")
        print("    Check dependencies and try again.\n")
        app = create_ui()
        app.launch(server_name="127.0.0.1", server_port=None)
        return

    print("\nInitializing rigger...")
    STATE.init_rigger()

    print("Pre-loading AI models (1-2 minutes on CPU)...")
    STATE.preload_models()

    print("\nLaunching UI...")
    app = create_ui()
    app.launch(server_name="127.0.0.1", server_port=None)


if __name__ == "__main__":
    launch_ui()
