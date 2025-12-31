#!/usr/bin/env python3
"""VEILBREAKERS RIGGER - PROFESSIONAL UI"""

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import json
import tempfile
import shutil
import os
from typing import Optional, List, Tuple, Dict, Any

# Import our rigger
from veilbreakers_rigger import (
    VeilbreakersRigger,
    BODY_TEMPLATES,
    InpaintQuality,
    ExportFormat,
    BodyPart,
    Point
)

# =============================================================================
# GLOBAL STATE
# =============================================================================

class AppState:
    """Application state manager"""

    def __init__(self):
        self.rigger: Optional[VeilbreakersRigger] = None
        self.mode = "select"
        self.current_part_name = ""
        self.history: List[np.ndarray] = []
        self.history_index = -1

    def init_rigger(self, sam_size: str = "large"):
        """Initialize or reinitialize the rigger"""
        try:
            self.rigger = VeilbreakersRigger(
                output_dir="./output",
                sam_size=sam_size,
                use_fallback=True
            )
            return True
        except Exception as e:
            print(f"Error initializing rigger: {e}")
            self.rigger = VeilbreakersRigger(
                output_dir="./output",
                sam_size="tiny",
                use_fallback=True
            )
            return False

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

STATE = AppState()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_visualization(show_mask: bool = True) -> Optional[np.ndarray]:
    """Create visualization of current state"""
    if STATE.rigger is None or STATE.rigger.current_rig is None:
        return None

    vis = STATE.rigger.get_working_image().copy()

    if show_mask and STATE.rigger.current_mask is not None:
        mask = STATE.rigger.current_mask
        overlay = np.zeros_like(vis)
        overlay[mask > 0] = [255, 100, 100]
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

# =============================================================================
# MAIN UI FUNCTIONS
# =============================================================================

def load_image(image, sam_size: str):
    """Load an image into the rigger"""
    if image is None:
        return None, "No image provided", [], gr.update(choices=[""]), None

    if STATE.rigger is None:
        STATE.init_rigger(sam_size)

    STATE.reset()

    try:
        if isinstance(image, str):
            STATE.rigger.load_image(image)
        else:
            STATE.rigger.load_image_array(image, "monster")

        STATE.save_state()

        vis = create_visualization(show_mask=False)
        parts = get_parts_table()
        choices = get_part_choices()

        return (
            vis,
            f"Loaded image ({STATE.rigger.current_rig.original_image.shape[1]}x{STATE.rigger.current_rig.original_image.shape[0]})",
            parts,
            gr.update(choices=choices),
            STATE.rigger.get_original_image()
        )
    except Exception as e:
        return None, f"Error: {str(e)}", [], gr.update(choices=[""]), None


def on_image_click(image, evt: gr.SelectData, mode: str):
    """Handle click on image"""
    if STATE.rigger is None or STATE.rigger.current_rig is None:
        return None, "Load an image first!"

    x, y = evt.index

    try:
        if mode == "select" or STATE.rigger.current_mask is None:
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


def auto_detect(prompt: str, box_thresh: float, text_thresh: float, quality: str):
    """Auto-detect parts from text prompt"""
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

    with gr.Blocks(
        title="VEILBREAKERS Monster Rigger",
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate")
    ) as app:

        gr.Markdown("# VEILBREAKERS Monster Rigger")
        gr.Markdown("Transform your monster art into fully-rigged, animated characters with AI-powered segmentation.")

        with gr.Row():
            # LEFT PANEL
            with gr.Column(scale=2):

                with gr.Tab("Workspace"):
                    main_image = gr.Image(
                        label="Monster Image (Upload or Drop Here)",
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

                with gr.Tab("Auto-Detect"):
                    gr.Markdown("### AI-Powered Detection")

                    preset_dropdown = gr.Dropdown(
                        choices=list(BODY_TEMPLATES.keys()),
                        value="quadruped",
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

                with gr.Tab("Manual Add"):
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

        # Hidden state
        original_image = gr.State(None)

        # EVENT HANDLERS
        load_btn.click(fn=load_image, inputs=[main_image, sam_size], outputs=[main_image, status_text, parts_table, parent_dropdown, original_image])
        main_image.select(fn=on_image_click, inputs=[main_image, mode_radio], outputs=[main_image, status_text])
        clear_btn.click(fn=clear_selection, outputs=[main_image, status_text])
        reset_btn.click(fn=reset_all, outputs=[main_image, status_text, parts_table, parent_dropdown, original_image])
        undo_btn.click(fn=undo_action, outputs=[main_image, status_text])
        redo_btn.click(fn=redo_action, outputs=[main_image, status_text])
        preset_dropdown.change(fn=get_preset_info, inputs=[preset_dropdown], outputs=[preset_info])
        apply_preset_btn.click(fn=apply_preset, inputs=[preset_dropdown, inpaint_quality], outputs=[main_image, status_text, parts_table, parent_dropdown])
        detect_btn.click(fn=auto_detect, inputs=[custom_prompt, box_thresh, text_thresh, inpaint_quality], outputs=[main_image, status_text, parts_table, parent_dropdown])
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
        export_btn.click(fn=export_rig, inputs=[monster_name, export_format], outputs=[export_status, download_file])
        app.load(fn=lambda: get_preset_info("quadruped"), outputs=[preset_info])

    return app


def launch_ui():
    """Launch the UI"""
    print("========================================================================")
    print("                    VEILBREAKERS MONSTER RIGGER")
    print("                         Launching UI...")
    print("========================================================================")

    STATE.init_rigger()

    app = create_ui()
    app.launch(server_name="127.0.0.1", server_port=None)


if __name__ == "__main__":
    launch_ui()
