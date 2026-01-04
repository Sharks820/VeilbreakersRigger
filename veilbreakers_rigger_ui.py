#!/usr/bin/env python3
"""VEILBREAKERS RIGGER - SIMPLIFIED UI v5.0

Super simple workflow:
1. Drop image → Smart Scan
2. See what's found → Click to add missing parts
3. Export with animations

No complex tabs. No confusing options. Just works.
"""

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
import tempfile
import shutil
import os
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

# =============================================================================
# IMPORTS WITH GRACEFUL DEGRADATION
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

# Animation system
ANIMATION_AVAILABLE = True
try:
    from spine_rig_builder import SpineRigBuilder, ARCHETYPE_CONFIGS, CreatureArchetype
    from animation_templates import AnimationTemplates
    ARCHETYPES = [a.name.lower().replace('_', ' ').title() for a in CreatureArchetype]
    ARCHETYPE_MAP = {a.name.lower().replace('_', ' ').title(): a.name.lower() for a in CreatureArchetype}
except Exception as e:
    ANIMATION_AVAILABLE = False
    ARCHETYPES = ["Humanoid", "Quadruped", "Winged", "Serpent", "Spider", "Eldritch"]
    ARCHETYPE_MAP = {a: a.lower() for a in ARCHETYPES}
    print(f"WARNING: Could not import animation system: {e}")

# =============================================================================
# GLOBAL STATE - MINIMAL
# =============================================================================

class AppState:
    """Minimal state - just what we need"""

    def __init__(self):
        self.rigger: Optional[VeilbreakersRigger] = None
        self.models_loaded = False
        self.original_image: Optional[np.ndarray] = None
        self.last_scan_parts: List[str] = []

    def init_rigger(self):
        """Initialize the rigger"""
        if not RIGGER_AVAILABLE:
            return False
        try:
            self.rigger = VeilbreakersRigger(
                output_dir="./output",
                sam_size="large",
                use_fallback=True
            )
            return True
        except Exception as e:
            print(f"Error: {e}")
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
        """Pre-load AI models"""
        if self.models_loaded or self.rigger is None:
            return
        try:
            self.rigger.segmenter.load()
            self.models_loaded = True
            print("AI models loaded!")
        except Exception as e:
            print(f"Model loading error: {e}")

STATE = AppState()

# =============================================================================
# VISUALIZATION - Always returns an image, NEVER None
# =============================================================================

def create_visualization(show_boxes: bool = True) -> np.ndarray:
    """Create visualization - ALWAYS returns valid image"""
    # If no rigger or no image, return a placeholder
    if STATE.rigger is None or STATE.rigger.current_rig is None:
        # Return a gray placeholder
        placeholder = np.full((400, 600, 3), 128, dtype=np.uint8)
        return placeholder

    try:
        vis = STATE.rigger.get_working_image()
        if vis is None:
            return np.full((400, 600, 3), 128, dtype=np.uint8)
        vis = vis.copy()
    except:
        return np.full((400, 600, 3), 128, dtype=np.uint8)

    # Show current selection mask if any
    if STATE.rigger.current_mask is not None:
        mask = STATE.rigger.current_mask
        overlay = np.zeros_like(vis)
        overlay[mask > 0] = (255, 100, 100)  # Red highlight
        vis = (vis * 0.7 + overlay * 0.3).astype(np.uint8)

    # Draw bounding boxes around detected parts
    if show_boxes:
        try:
            pil_img = Image.fromarray(vis)
            draw = ImageDraw.Draw(pil_img)

            parts = STATE.rigger.get_parts()
            for part in parts:
                bbox = None
                if hasattr(part, 'bbox') and part.bbox:
                    bbox = (part.bbox.x1, part.bbox.y1, part.bbox.x2, part.bbox.y2)
                elif hasattr(part, 'bounds') and part.bounds:
                    bbox = part.bounds
                elif hasattr(part, 'mask') and part.mask is not None:
                    # Calculate bbox from mask
                    ys, xs = np.where(part.mask > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

                if bbox:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    # GREEN box
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                    # Label
                    label = part.name
                    try:
                        draw.rectangle([x1, y1-18, x1+len(label)*8+4, y1], fill=(0, 255, 0))
                        draw.text((x1+2, y1-16), label, fill=(0, 0, 0))
                    except:
                        draw.text((x1, y1-15), label, fill=(0, 255, 0))

            vis = np.array(pil_img)
        except Exception as e:
            print(f"Box drawing error: {e}")

    return vis

def get_parts_summary() -> str:
    """Get a simple summary of detected parts"""
    if STATE.rigger is None:
        return "No image loaded"

    parts = STATE.rigger.get_parts()
    if not parts:
        return "No parts detected yet. Click 'Smart Scan' to auto-detect."

    part_names = [p.name for p in parts]
    STATE.last_scan_parts = part_names

    # Group by type
    heads = [p for p in part_names if 'head' in p.lower() or 'face' in p.lower()]
    arms = [p for p in part_names if 'arm' in p.lower() or 'hand' in p.lower() or 'claw' in p.lower()]
    legs = [p for p in part_names if 'leg' in p.lower() or 'foot' in p.lower() or 'paw' in p.lower()]
    body = [p for p in part_names if 'body' in p.lower() or 'torso' in p.lower() or 'chest' in p.lower()]
    extras = [p for p in part_names if p not in heads + arms + legs + body]

    lines = [f"**Found {len(parts)} parts:**\n"]
    if heads: lines.append(f"Head: {', '.join(heads)}")
    if body: lines.append(f"Body: {', '.join(body)}")
    if arms: lines.append(f"Arms/Hands: {', '.join(arms)}")
    if legs: lines.append(f"Legs/Feet: {', '.join(legs)}")
    if extras: lines.append(f"Other: {', '.join(extras)}")

    lines.append("\n**Missing something?** Click on the image to select, then name it below.")

    return "\n".join(lines)

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_and_scan(image):
    """Load image AND run smart scan in one step"""
    if image is None:
        return create_visualization(), "Please upload an image first", ""

    # Initialize rigger if needed
    if STATE.rigger is None:
        if not STATE.init_rigger():
            return create_visualization(), "ERROR: Could not initialize AI models", ""

    # Load the image
    try:
        if isinstance(image, np.ndarray):
            np_image = image
        else:
            np_image = np.array(image)

        STATE.original_image = np_image.copy()
        STATE.rigger.load_image(np_image)

        # Pre-load models if not done
        if not STATE.models_loaded:
            STATE.preload_models()

    except Exception as e:
        return create_visualization(), f"Error loading image: {e}", ""

    # Run smart detection
    try:
        parts = STATE.rigger.smart_detect(
            use_florence=True,
            box_threshold=0.15,
            extract_parts=True,
            inpaint_quality=InpaintQuality.STANDARD
        )

        vis = create_visualization(show_boxes=True)
        summary = get_parts_summary()

        if parts:
            status = f"Found {len(parts)} parts! Green boxes show detected regions."
        else:
            status = "No parts auto-detected. Click on the image to manually select regions."

        return vis, status, summary

    except Exception as e:
        import traceback
        traceback.print_exc()
        return create_visualization(), f"Detection error: {e}", get_parts_summary()


def on_image_click(image, evt: gr.SelectData):
    """Handle click - select region at click point"""
    if STATE.rigger is None or STATE.rigger.current_rig is None:
        # CRITICAL: Return the image, not None!
        if image is not None:
            return image, "Load an image first (click 'Smart Scan')"
        return create_visualization(), "Load an image first"

    try:
        x, y = evt.index

        # Make sure segmenter has the image
        working = STATE.rigger.get_working_image()
        if working is not None:
            STATE.rigger.segmenter.set_image(working)

        # Click to segment
        STATE.rigger.click_segment(x, y)

        vis = create_visualization()
        return vis, f"Selected region at ({x}, {y}). Name it below and click 'Add This Part'."

    except Exception as e:
        # CRITICAL: Return current visualization, not None!
        return create_visualization(), f"Click error: {e}"


def add_missing_part(part_name: str):
    """Add the currently selected region as a named part"""
    if STATE.rigger is None:
        return create_visualization(), "Load an image first", get_parts_summary()

    if not part_name or not part_name.strip():
        return create_visualization(), "Please enter a part name", get_parts_summary()

    if STATE.rigger.current_mask is None:
        return create_visualization(), "Click on the image first to select a region", get_parts_summary()

    try:
        # Add the part
        part = STATE.rigger.add_part(
            name=part_name.strip(),
            z_index=len(STATE.rigger.get_parts()),
            inpaint_quality=InpaintQuality.STANDARD
        )

        vis = create_visualization(show_boxes=True)
        summary = get_parts_summary()

        return vis, f"Added '{part_name}' successfully!", summary

    except Exception as e:
        return create_visualization(), f"Error adding part: {e}", get_parts_summary()


def export_rig(monster_name: str, archetype: str):
    """Build skeleton, generate animations, export everything"""
    if STATE.rigger is None or len(STATE.rigger.get_parts()) == 0:
        return "⚠️ No parts detected! Run Smart Scan first.", None

    if not monster_name or not monster_name.strip():
        monster_name = "monster"

    monster_name = monster_name.strip().replace(" ", "_")

    try:
        # Create temp directory for export
        temp_dir = Path(tempfile.mkdtemp())

        # Export as Spine JSON with animations
        if ANIMATION_AVAILABLE:
            # Get archetype key
            arch_key = ARCHETYPE_MAP.get(archetype, "humanoid")

            # Get parts info
            parts = STATE.rigger.get_parts()
            part_names = [p.name.lower() for p in parts]

            # Detect creature features from parts
            has_tail = any('tail' in p for p in part_names)
            has_wings = any('wing' in p for p in part_names)
            arm_count = len([p for p in part_names if 'arm' in p])
            leg_count = len([p for p in part_names if 'leg' in p])

            # BUILD THE SKELETON
            builder = SpineRigBuilder()
            spine_data = builder.build_rig(
                parts=parts,
                archetype=arch_key,
                rig_name=monster_name,
                arm_count=max(2, arm_count),
                leg_count=max(2, leg_count),
                has_tail=has_tail,
                has_wings=has_wings
            )

            # Count what we built
            bone_count = len(spine_data.get('bones', []))
            slot_count = len(spine_data.get('slots', []))
            anim_count = len(spine_data.get('animations', {}))
            anim_names = list(spine_data.get('animations', {}).keys())

            # Save Spine JSON
            output_path = temp_dir / f"{monster_name}_spine.json"
            with open(output_path, 'w') as f:
                json.dump(spine_data, f, indent=2)

            # Export part images as PNGs
            parts_dir = temp_dir / "parts"
            parts_dir.mkdir(exist_ok=True)

            exported_parts = 0
            for part in parts:
                if hasattr(part, 'image') and part.image is not None:
                    part_img = Image.fromarray(part.image)
                    part_img.save(parts_dir / f"{part.name}.png")
                    exported_parts += 1

            # Create zip
            zip_path = temp_dir / f"{monster_name}_rig"
            shutil.make_archive(str(zip_path), 'zip', temp_dir)

            # Build detailed status message
            status_lines = [
                f"✅ BUILT: {monster_name}",
                f"🦴 Skeleton: {bone_count} bones, {slot_count} slots",
                f"🎬 Animations: {anim_count} ({', '.join(anim_names[:5])}{'...' if len(anim_names) > 5 else ''})",
                f"📁 Parts: {exported_parts} PNG images",
                f"",
                f"Ready for Godot! Import the Spine JSON."
            ]

            return "\n".join(status_lines), str(zip_path) + ".zip"
        else:
            # Basic PNG export (no animation system)
            parts_dir = temp_dir / "parts"
            parts_dir.mkdir(exist_ok=True)

            for part in STATE.rigger.get_parts():
                if hasattr(part, 'image') and part.image is not None:
                    part_img = Image.fromarray(part.image)
                    part_img.save(parts_dir / f"{part.name}.png")

            zip_path = temp_dir / f"{monster_name}_parts"
            shutil.make_archive(str(zip_path), 'zip', parts_dir)

            return f"📁 Exported {len(STATE.rigger.get_parts())} parts as PNGs\n(Animation system not loaded)", str(zip_path) + ".zip"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Build error: {e}", None


def clear_all():
    """Reset everything"""
    if STATE.rigger:
        STATE.rigger.reset()
    STATE.original_image = None
    STATE.last_scan_parts = []
    return create_visualization(), "Cleared. Upload a new image to start.", ""


def get_archetype_preview(archetype: str) -> str:
    """Show what animations will be generated for this archetype"""
    if not ANIMATION_AVAILABLE:
        return "Animation system not loaded"

    arch_key = ARCHETYPE_MAP.get(archetype, "humanoid")

    # Get the config for this archetype
    try:
        from spine_rig_builder import ARCHETYPE_CONFIGS
        config = ARCHETYPE_CONFIGS.get(arch_key, {})
        anims = config.get("animations", [])

        if not anims:
            return f"**{archetype}**: Default animations (idle, walk, attack)"

        # Group animations by type
        lines = [f"**{archetype} Animations ({len(anims)} total):**"]

        # Categorize
        idles = [a for a in anims if 'idle' in a.lower()]
        walks = [a for a in anims if 'walk' in a.lower() or 'run' in a.lower() or 'move' in a.lower()]
        attacks = [a for a in anims if 'attack' in a.lower() or 'strike' in a.lower() or 'bite' in a.lower()]
        others = [a for a in anims if a not in idles + walks + attacks]

        if idles: lines.append(f"• Idle: {', '.join(idles[:3])}{'...' if len(idles) > 3 else ''}")
        if walks: lines.append(f"• Movement: {', '.join(walks[:3])}{'...' if len(walks) > 3 else ''}")
        if attacks: lines.append(f"• Combat: {', '.join(attacks[:3])}{'...' if len(attacks) > 3 else ''}")
        if others: lines.append(f"• Special: {', '.join(others[:3])}{'...' if len(others) > 3 else ''}")

        return "\n".join(lines)
    except Exception as e:
        return f"**{archetype}**: Standard monster animations"


# =============================================================================
# THE UI - DEAD SIMPLE
# =============================================================================

def create_ui():
    """Create the simplified UI"""

    if not RIGGER_AVAILABLE:
        with gr.Blocks(title="VEILBREAKERS - ERROR") as app:
            gr.Markdown("# VEILBREAKERS Monster Rigger")
            gr.Markdown("## ERROR: Could not load AI models")
            gr.Markdown("Check console for details.")
        return app

    with gr.Blocks(
        title="VEILBREAKERS Monster Rigger",
        theme=gr.themes.Soft(primary_hue="orange")
    ) as app:

        gr.Markdown("# VEILBREAKERS Monster Rigger v5.0")
        gr.Markdown("*Drop image → Smart Scan → Add missing parts → Export*")

        with gr.Row():
            # LEFT: The image
            with gr.Column(scale=2):
                main_image = gr.Image(
                    label="Monster Image",
                    type="numpy",
                    height=600,
                    interactive=True,
                    sources=["upload", "clipboard"]
                )

                with gr.Row():
                    scan_btn = gr.Button("SMART SCAN", variant="primary", size="lg", scale=2)
                    clear_btn = gr.Button("Clear", variant="stop", size="lg", scale=1)

                status = gr.Textbox(label="Status", interactive=False, lines=2)

            # RIGHT: Controls
            with gr.Column(scale=1):
                # Parts found
                gr.Markdown("## Parts Found")
                parts_display = gr.Markdown("Upload an image and click 'Smart Scan'")

                gr.Markdown("---")

                # Add missing part
                gr.Markdown("## Add Missing Part")
                gr.Markdown("*Click on the image to select a region, then name it:*")

                part_name_input = gr.Textbox(
                    label="Part Name",
                    placeholder="e.g., left_arm, tail, horn",
                    info="Click image first to select the region"
                )
                add_btn = gr.Button("Add This Part", variant="secondary")

                gr.Markdown("---")

                # RIG & ANIMATE - Make it OBVIOUS this is where the magic happens
                gr.Markdown("## 🦴 RIG & ANIMATE")
                gr.Markdown("*This builds the skeleton + generates all animations!*")

                monster_name = gr.Textbox(label="Monster Name", placeholder="my_monster")
                archetype = gr.Dropdown(
                    choices=ARCHETYPES,
                    value=ARCHETYPES[0] if ARCHETYPES else "Humanoid",
                    label="Creature Type (determines animations)"
                )

                # Show what animations will be generated
                anim_preview = gr.Markdown("*Select creature type to see animations*")

                export_btn = gr.Button("🎬 BUILD RIG + ANIMATIONS", variant="primary", size="lg")
                export_status = gr.Textbox(label="Build Status", interactive=False, lines=3)
                download = gr.File(label="Download Spine JSON + Parts")

        # EVENT HANDLERS - Simple!

        # Smart Scan: Load AND detect in one click
        scan_btn.click(
            fn=load_and_scan,
            inputs=[main_image],
            outputs=[main_image, status, parts_display]
        )

        # Click to select region - MUST preserve image
        main_image.select(
            fn=on_image_click,
            inputs=[main_image],
            outputs=[main_image, status]
        )

        # Add missing part
        add_btn.click(
            fn=add_missing_part,
            inputs=[part_name_input],
            outputs=[main_image, status, parts_display]
        )

        # Clear
        clear_btn.click(
            fn=clear_all,
            outputs=[main_image, status, parts_display]
        )

        # Archetype selection shows what animations you'll get
        archetype.change(
            fn=get_archetype_preview,
            inputs=[archetype],
            outputs=[anim_preview]
        )

        # BUILD RIG + ANIMATIONS
        export_btn.click(
            fn=export_rig,
            inputs=[monster_name, archetype],
            outputs=[export_status, download]
        )

        # Startup - show initial archetype preview
        app.load(
            fn=lambda: ("Ready! Upload an image and click 'Smart Scan'", get_archetype_preview(ARCHETYPES[0] if ARCHETYPES else "Humanoid")),
            outputs=[status, anim_preview]
        )

    return app


# =============================================================================
# ENTRY POINT
# =============================================================================

def launch_ui():
    """Launch the UI - called by run.py"""
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    launch_ui()
