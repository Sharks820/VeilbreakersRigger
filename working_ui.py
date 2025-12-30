"""VEILBREAKERS Rigger - Working UI v2"""
import gradio as gr
import numpy as np

# Import rigger
from veilbreakers_rigger import VeilbreakersRigger, BODY_TEMPLATES, InpaintQuality

# Global state
RIGGER = None
MODELS_LOADED = False
CURRENT_IMAGE = None  # Track the original image

def init():
    """Initialize rigger"""
    global RIGGER
    if RIGGER is None:
        RIGGER = VeilbreakersRigger(output_dir="./output", sam_size="large", use_fallback=True)

def preload_models():
    """Pre-load AI models at startup"""
    global RIGGER, MODELS_LOADED
    if MODELS_LOADED:
        return
    init()
    print("Pre-loading AI models (this takes 1-2 minutes on CPU)...")
    try:
        RIGGER.segmenter.load()
        MODELS_LOADED = True
        print("AI models loaded successfully!")
    except Exception as e:
        print(f"Warning: Model pre-load failed: {e}")

def load_new_image(image):
    """Load a new image - always resets first"""
    global RIGGER, CURRENT_IMAGE

    if image is None:
        return None, "Upload an image first"

    init()

    try:
        # Always reset when loading new image
        RIGGER.reset()
        CURRENT_IMAGE = image.copy()

        # Load into rigger
        RIGGER.load_image_array(image, "monster")

        # Set image in segmenter for click-to-segment
        RIGGER.segmenter.set_image(image)

        # Don't auto-detect - let user click "Smart Detect" when ready
        return image, f"Loaded {image.shape[1]}x{image.shape[0]}. Click 'Smart Detect (Florence-2)' to find body parts."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return image, f"Error loading: {str(e)}"

def auto_detect_parts(image, prompt, threshold):
    """Auto-detect parts from text prompt"""
    global RIGGER, MODELS_LOADED

    if image is None:
        return None, "Upload and load an image first"

    if RIGGER is None or RIGGER.current_rig is None:
        return image, "Click 'Load Image' first"

    if not MODELS_LOADED:
        return image, "AI models still loading... wait a moment and try again"

    if not prompt or not prompt.strip():
        return image, "Enter a detection prompt like 'head . body . arms . legs'"

    try:
        print(f"Auto-detecting: {prompt} (threshold={threshold})")
        parts = RIGGER.auto_detect(
            prompt,
            box_threshold=threshold,
            text_threshold=threshold,
            inpaint_quality=InpaintQuality.STANDARD
        )

        if len(parts) == 0:
            return image, f"No parts detected. Try lowering threshold or simpler terms like 'body . head . arm'"

        vis = RIGGER.get_working_image()
        part_names = [p.name for p in parts]
        return vis, f"Detected {len(parts)} parts: {', '.join(part_names)}. Click 'Export' when ready."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return image, f"Detection error: {str(e)}"

def smart_detect_parts(image, prompt, threshold):
    """
    BEST detection - uses Florence-2 unified vision model.
    No prompt required! Florence-2 finds AND locates all parts automatically.
    """
    global RIGGER, MODELS_LOADED

    if image is None:
        return None, "Upload and load an image first"

    if RIGGER is None or RIGGER.current_rig is None:
        return image, "Click 'Load Image' first"

    if not MODELS_LOADED:
        return image, "AI models still loading... wait a moment and try again"

    try:
        print(f"Smart-detecting with Florence-2 (threshold={threshold})")

        # Use smart_detect - Florence-2 first, then Grounding DINO fallback
        text_prompt = prompt.strip() if prompt and prompt.strip() else None
        parts = RIGGER.smart_detect(
            text_prompt=text_prompt,
            use_florence=True,  # Florence-2 primary
            box_threshold=threshold,
            inpaint_quality=InpaintQuality.STANDARD
        )

        if len(parts) == 0:
            return image, "No parts detected. Try 'Find All Segments' or click directly on the image."

        vis = RIGGER.get_working_image()
        part_names = [p.name for p in parts]
        return vis, f"Detected {len(parts)} parts: {', '.join(part_names)}. Click 'Export' when ready."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return image, f"Detection error: {str(e)}"

def redetect_single_part(image, part_prompt, threshold):
    """Re-detect a single part with custom settings"""
    global RIGGER, MODELS_LOADED

    if image is None:
        return None, "Upload and load an image first"

    if RIGGER is None or RIGGER.current_rig is None:
        return image, "Click 'Load Image' first"

    if not MODELS_LOADED:
        return image, "AI models still loading..."

    if not part_prompt or not part_prompt.strip():
        return image, "Enter a single part name like 'head' or 'arm'"

    try:
        # Detect single part from ORIGINAL image
        print(f"Re-detecting: {part_prompt} (threshold={threshold})")

        # Set segmenter to original image for fresh detection
        RIGGER.segmenter.set_image(RIGGER.current_rig.original_image)

        detections = RIGGER.segmenter.auto_detect(
            part_prompt.strip(),
            box_threshold=threshold,
            text_threshold=threshold
        )

        if not detections:
            return image, f"'{part_prompt}' not detected. Try lower threshold or different term."

        # Take the best detection
        name, mask, confidence = detections[0]
        print(f"Found {name} with confidence {confidence:.2f}")

        # Show mask overlay on original
        vis = RIGGER.current_rig.original_image.copy()
        overlay = np.zeros_like(vis)
        overlay[mask > 0] = [50, 255, 50]  # Green overlay
        vis = (vis.astype(float) * 0.6 + overlay.astype(float) * 0.4).astype(np.uint8)

        # Store mask for adding
        RIGGER.current_mask = mask

        return vis, f"Found '{name}' (confidence: {confidence:.1%}). Enter name and click 'Add Part' to save."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return image, f"Re-detect error: {str(e)}"

def click_segment(image, evt: gr.SelectData):
    """Handle click to segment at a point"""
    global RIGGER, CURRENT_IMAGE

    if image is None:
        return image, "Upload an image first"

    if RIGGER is None or RIGGER.current_rig is None:
        return image, "Click 'Load Image' first"

    try:
        x, y = evt.index
        print(f"Click at ({x}, {y})")

        # Make sure segmenter has the working image
        working = RIGGER.get_working_image()
        if working is not None:
            RIGGER.segmenter.set_image(working)

        # Segment at click point
        RIGGER.click_segment(x, y)

        # Show selection overlay
        vis = working.copy() if working is not None else image.copy()
        if RIGGER.current_mask is not None:
            mask = RIGGER.current_mask
            # Red overlay for selection
            overlay = np.zeros_like(vis)
            overlay[mask > 0] = [255, 50, 50]
            vis = (vis.astype(float) * 0.6 + overlay.astype(float) * 0.4).astype(np.uint8)

        return vis, f"Selected region at ({x}, {y}). Enter part name and click 'Add Part'."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return image, f"Click error: {str(e)}"

def add_current_part(image, name, z_index):
    """Add the current selection as a named part"""
    global RIGGER

    if RIGGER is None or RIGGER.current_mask is None:
        return image, "No selection. Click on image first to select a region."

    if not name or not name.strip():
        return image, "Enter a part name (e.g., 'head', 'body', 'arm')"

    try:
        RIGGER.add_part(name=name.strip(), z_index=int(z_index), inpaint_quality=InpaintQuality.STANDARD)
        vis = RIGGER.get_working_image()
        parts = RIGGER.get_parts()

        # Update segmenter with new working image (part removed)
        if vis is not None:
            RIGGER.segmenter.set_image(vis)

        return vis, f"Added '{name}'. Total parts: {len(parts)}. Continue selecting or export."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return image, f"Add part error: {str(e)}"

def get_saved_parts():
    """Get list of currently saved parts"""
    global RIGGER

    if RIGGER is None or RIGGER.current_rig is None:
        return "No parts saved yet. Load an image and detect parts first."

    parts = RIGGER.get_parts()
    if not parts:
        return "No parts saved yet."

    lines = [f"**{len(parts)} parts saved:**"]
    for i, p in enumerate(parts):
        lines.append(f"{i+1}. **{p.name}** (z={p.z_index})")

    return "\n".join(lines)

def remove_part(part_name):
    """Remove a saved part so it can be re-detected"""
    global RIGGER

    if RIGGER is None or RIGGER.current_rig is None:
        return None, "No image loaded"

    if not part_name or not part_name.strip():
        return None, "Enter the part name to remove"

    name = part_name.strip()
    parts = RIGGER.get_parts()
    part_names = [p.name for p in parts]

    if name not in part_names:
        return None, f"Part '{name}' not found. Saved parts: {', '.join(part_names)}"

    try:
        RIGGER.remove_part(name)
        remaining = RIGGER.get_parts()

        # Show original image for re-detection
        vis = RIGGER.current_rig.original_image.copy()

        return vis, f"Removed '{name}'. {len(remaining)} parts remaining. You can now re-detect it."
    except Exception as e:
        return None, f"Error removing part: {str(e)}"

def export_monster(name):
    """Export the rig to files"""
    global RIGGER

    if RIGGER is None:
        return "No rigger initialized"

    parts = RIGGER.get_parts() if RIGGER.current_rig else []
    if not parts:
        return "No parts to export. Use Auto-Detect or click to add parts first."

    try:
        export_name = name.strip() if name and name.strip() else "monster"
        path = RIGGER.export(name=export_name)
        part_names = [p.name for p in parts]
        return f"SUCCESS! Exported to:\n{path}\n\nParts ({len(parts)}): {', '.join(part_names)}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Export error: {str(e)}"

def full_reset():
    """Complete reset - ready for new monster"""
    global RIGGER, CURRENT_IMAGE

    if RIGGER is not None:
        RIGGER.reset()
    CURRENT_IMAGE = None

    return None, "Reset complete. Upload a new image to start."

# Track segment browsing state
SEGMENT_COUNT = 0
CURRENT_SEG_INDEX = 0

def segment_everything(image):
    """Find all segments in the image automatically"""
    global RIGGER, SEGMENT_COUNT, CURRENT_SEG_INDEX

    if image is None:
        return image, "Upload and load an image first", 0, 0

    if RIGGER is None or RIGGER.current_rig is None:
        return image, "Click 'Load Image' first", 0, 0

    try:
        print("Finding all segments...")
        segments = RIGGER.segment_everything(min_mask_area=500)
        SEGMENT_COUNT = len(segments)
        CURRENT_SEG_INDEX = 0

        if SEGMENT_COUNT == 0:
            return image, "No segments found. Try clicking directly on the image.", 0, 0

        # Show first segment
        vis = RIGGER.get_segment_preview(0)
        RIGGER.select_segment(0)

        return vis, f"Found {SEGMENT_COUNT} segments! Use Prev/Next to browse, then 'Add Selected Part' to save.", SEGMENT_COUNT, 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        return image, f"Segmentation error: {str(e)}", 0, 0

def show_segment(index):
    """Show a specific segment by index"""
    global RIGGER, SEGMENT_COUNT, CURRENT_SEG_INDEX

    if RIGGER is None or RIGGER.current_rig is None:
        return None, "No image loaded", 0

    if SEGMENT_COUNT == 0:
        return None, "Run 'Find All Segments' first", 0

    # Clamp index
    index = max(0, min(int(index), SEGMENT_COUNT - 1))
    CURRENT_SEG_INDEX = index

    try:
        vis = RIGGER.get_segment_preview(index)
        RIGGER.select_segment(index)
        mask = RIGGER.current_mask
        area = (mask > 0).sum() if mask is not None else 0
        return vis, f"Segment {index + 1}/{SEGMENT_COUNT} (area: {area:,} px). Name it and click 'Add Selected Part'.", index
    except Exception as e:
        return None, f"Error: {str(e)}", index

def prev_segment():
    """Show previous segment"""
    global CURRENT_SEG_INDEX, SEGMENT_COUNT
    new_index = max(0, CURRENT_SEG_INDEX - 1)
    return show_segment(new_index)

def next_segment():
    """Show next segment"""
    global CURRENT_SEG_INDEX, SEGMENT_COUNT
    new_index = min(SEGMENT_COUNT - 1, CURRENT_SEG_INDEX + 1)
    return show_segment(new_index)

def handle_box_select(image, evt: gr.SelectData, box_mode_enabled):
    """Handle box selection when box mode is enabled"""
    global RIGGER

    if not box_mode_enabled:
        # Regular click-to-segment
        return click_segment(image, evt)

    # For box mode, we need start and end points
    # Gradio doesn't support drag selection natively, so we'll use two clicks
    # Store first click, segment on second click
    if not hasattr(handle_box_select, 'first_click'):
        handle_box_select.first_click = None

    if handle_box_select.first_click is None:
        # First click - store it
        handle_box_select.first_click = evt.index
        return image, f"Box start: ({evt.index[0]}, {evt.index[1]}). Click again for end point."
    else:
        # Second click - create box
        x1, y1 = handle_box_select.first_click
        x2, y2 = evt.index
        handle_box_select.first_click = None

        if RIGGER is None or RIGGER.current_rig is None:
            return image, "Load image first"

        try:
            RIGGER.box_segment(x1, y1, x2, y2)
            vis = RIGGER.current_rig.original_image.copy()
            if RIGGER.current_mask is not None:
                overlay = np.zeros_like(vis)
                overlay[RIGGER.current_mask > 0] = [255, 50, 50]
                vis = (vis.astype(float) * 0.6 + overlay.astype(float) * 0.4).astype(np.uint8)
            return vis, f"Box segment: ({x1},{y1}) to ({x2},{y2}). Name it and click 'Add Selected Part'."
        except Exception as e:
            return image, f"Box segment error: {str(e)}"

# Build UI
with gr.Blocks(title="VEILBREAKERS Rigger", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# VEILBREAKERS Monster Rigger v3.0")
    gr.Markdown("*AI-powered monster segmentation for Godot*")

    with gr.Row():
        # Left column - Image
        with gr.Column(scale=2):
            image = gr.Image(
                label="Monster Image (click to select parts)",
                type="numpy",
                height=512,
                sources=["upload", "clipboard"]
            )
            status = gr.Textbox(label="Status", interactive=False, lines=2)

        # Right column - Controls
        with gr.Column(scale=1):
            gr.Markdown("### 1. Load Image")
            with gr.Row():
                load_btn = gr.Button("Load Image", variant="primary", size="lg")
                reset_btn = gr.Button("Reset", variant="stop")

            gr.Markdown("---")
            gr.Markdown("### 2. Smart Detection (Best)")
            gr.Markdown("*Florence-2 finds & locates all parts - NO PROMPT NEEDED*")

            with gr.Row():
                smart_detect_btn = gr.Button("Smart Detect (Florence-2)", variant="primary", size="lg")

            with gr.Row():
                segment_all_btn = gr.Button("Find All Segments", variant="secondary")
                segment_count = gr.Number(label="Found", value=0, interactive=False, scale=1)

            with gr.Row():
                prev_seg_btn = gr.Button("< Prev", size="sm")
                seg_index = gr.Number(label="Segment #", value=0, minimum=0, step=1, scale=1)
                next_seg_btn = gr.Button("Next >", size="sm")

            gr.Markdown("*Click on image to select manually, or enable box mode:*")
            box_mode = gr.Checkbox(label="Box Selection Mode (2-click)", value=False, info="Click two corners to draw box")

            gr.Markdown("---")
            gr.Markdown("### 3. Text Detection (Backup)")
            gr.Markdown("*Use if Smart Segmentation misses something*")
            prompt = gr.Textbox(
                label="Detection Prompt",
                placeholder="head . body . arms . legs",
                value="head . body . arms . legs",
                info="Separate parts with ' . '"
            )
            threshold = gr.Slider(
                0.1, 0.5, value=0.2, step=0.05,
                label="Threshold (lower = more sensitive)"
            )
            with gr.Row():
                detect_btn = gr.Button("Detect All", variant="secondary")
                single_part = gr.Textbox(label="Single Part", placeholder="e.g., head", scale=2)
                redetect_btn = gr.Button("Detect One", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("### 4. Save Part")
            part_name = gr.Textbox(label="Part Name", placeholder="e.g., head")
            z_index = gr.Slider(0, 10, value=0, step=1, label="Z-Index (back=0, front=10)")
            add_btn = gr.Button("Add Selected Part")

            gr.Markdown("---")
            gr.Markdown("### 5. Saved Parts")
            saved_parts_display = gr.Markdown("No parts saved yet.")
            refresh_parts_btn = gr.Button("Refresh Parts List", size="sm")
            with gr.Row():
                remove_part_name = gr.Textbox(label="Part to Remove", placeholder="e.g., head", scale=2)
                remove_btn = gr.Button("Remove", variant="stop", scale=1)

            gr.Markdown("---")
            gr.Markdown("### 6. Export")
            monster_name = gr.Textbox(label="Monster Name", value="monster")
            export_btn = gr.Button("Export to Godot", variant="primary", size="lg")
            export_status = gr.Textbox(label="Export Result", interactive=False, lines=3)

    # Event handlers
    load_btn.click(
        fn=load_new_image,
        inputs=[image],
        outputs=[image, status]
    )

    reset_btn.click(
        fn=full_reset,
        inputs=[],
        outputs=[image, status]
    )

    # PRIMARY: Smart Detect (RAM++) - uses auto-tagging, no prompt needed
    smart_detect_btn.click(
        fn=smart_detect_parts,
        inputs=[image, prompt, threshold],
        outputs=[image, status]
    )

    # BACKUP: Text-based detection
    detect_btn.click(
        fn=auto_detect_parts,
        inputs=[image, prompt, threshold],
        outputs=[image, status]
    )

    redetect_btn.click(
        fn=redetect_single_part,
        inputs=[image, single_part, threshold],
        outputs=[image, status]
    )

    add_btn.click(
        fn=add_current_part,
        inputs=[image, part_name, z_index],
        outputs=[image, status]
    )

    refresh_parts_btn.click(
        fn=get_saved_parts,
        inputs=[],
        outputs=[saved_parts_display]
    )

    remove_btn.click(
        fn=remove_part,
        inputs=[remove_part_name],
        outputs=[image, status]
    )

    export_btn.click(
        fn=export_monster,
        inputs=[monster_name],
        outputs=[export_status]
    )

    # Smart Segmentation handlers
    segment_all_btn.click(
        fn=segment_everything,
        inputs=[image],
        outputs=[image, status, segment_count, seg_index]
    )

    prev_seg_btn.click(
        fn=prev_segment,
        inputs=[],
        outputs=[image, status, seg_index]
    )

    next_seg_btn.click(
        fn=next_segment,
        inputs=[],
        outputs=[image, status, seg_index]
    )

    seg_index.change(
        fn=show_segment,
        inputs=[seg_index],
        outputs=[image, status, seg_index]
    )

    # Override image click to support box mode
    image.select(
        fn=handle_box_select,
        inputs=[image, box_mode],
        outputs=[image, status]
    )

if __name__ == "__main__":
    print("=" * 50)
    print("  VEILBREAKERS Monster Rigger v3.0")
    print("=" * 50)
    print()
    print("Pre-loading AI models (1-2 minutes on CPU)...")
    preload_models()
    print()
    print("Launching UI...")
    demo.launch(server_name="127.0.0.1")
