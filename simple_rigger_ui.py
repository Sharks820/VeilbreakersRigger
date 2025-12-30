"""VEILBREAKERS Rigger - Simple Working UI for Gradio 6.x"""
import gradio as gr
import numpy as np

# Try to import the rigger
try:
    from veilbreakers_rigger import VeilbreakersRigger, BODY_TEMPLATES
    RIGGER_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load rigger: {e}")
    RIGGER_AVAILABLE = False

# Global state
STATE = {"rigger": None, "image": None}

def init_rigger():
    if not RIGGER_AVAILABLE:
        return False
    try:
        STATE["rigger"] = VeilbreakersRigger(
            output_dir="./output",
            sam_size="large",
            use_fallback=True
        )
        return True
    except Exception as e:
        print(f"Error init rigger: {e}")
        return False

def process_upload(image):
    """Process uploaded image"""
    if image is None:
        return None, "Upload an image to get started"

    STATE["image"] = image
    h, w = image.shape[:2]

    if STATE["rigger"] is None:
        init_rigger()

    if STATE["rigger"]:
        try:
            STATE["rigger"].load_image_array(image, "monster")
            return image, f"Loaded {w}x{h} - Click on image to select parts"
        except Exception as e:
            return image, f"Loaded {w}x{h} - Error: {e}"

    return image, f"Loaded {w}x{h}"

def handle_click(image, evt: gr.SelectData):
    """Handle click"""
    if image is None or STATE["rigger"] is None:
        return image, "Upload image first"

    x, y = evt.index
    try:
        STATE["rigger"].click_segment(x, y)
        vis = STATE["rigger"].get_working_image()
        if STATE["rigger"].current_mask is not None:
            mask = STATE["rigger"].current_mask
            overlay = vis.copy()
            overlay[mask > 0] = [255, 100, 100]
            vis = (vis * 0.6 + overlay * 0.4).astype(np.uint8)
        return vis, f"Selected at ({x}, {y})"
    except Exception as e:
        return image, f"Error: {e}"

def detect_parts(image, prompt):
    """Auto detect"""
    if image is None or STATE["rigger"] is None:
        return image, "Upload image first"
    if not prompt:
        return image, "Enter prompt like: head . body . arms"
    try:
        parts = STATE["rigger"].auto_detect(prompt)
        return STATE["rigger"].get_working_image(), f"Found {len(parts)} parts"
    except Exception as e:
        return image, f"Error: {e}"

def reset_all():
    if STATE["rigger"]:
        STATE["rigger"].reset()
    STATE["image"] = None
    return None, "Reset - upload new image"

# Build UI
with gr.Blocks(title="VEILBREAKERS Rigger") as demo:
    gr.Markdown("# VEILBREAKERS Monster Rigger v3.0")

    with gr.Row():
        with gr.Column():
            img = gr.Image(
                label="Drop image here or click to upload",
                type="numpy",
                height=500
            )
            status = gr.Textbox(label="Status", value="Upload an image to start")
            reset_btn = gr.Button("Reset")

        with gr.Column():
            prompt = gr.Textbox(label="Auto-Detect Prompt", placeholder="head . body . arms . legs")
            detect_btn = gr.Button("Auto Detect Parts", variant="primary")
            gr.Markdown("---")
            gr.Markdown("### Instructions")
            gr.Markdown("1. Upload/drop an image")
            gr.Markdown("2. Click to select parts OR use auto-detect")
            gr.Markdown("3. Export when done")

    # Wire events
    img.change(process_upload, inputs=[img], outputs=[img, status])
    img.select(handle_click, inputs=[img], outputs=[img, status])
    detect_btn.click(detect_parts, inputs=[img, prompt], outputs=[img, status])
    reset_btn.click(reset_all, outputs=[img, status])

def launch():
    print("Starting VEILBREAKERS Rigger...")
    init_rigger()
    demo.launch(server_name="127.0.0.1")

if __name__ == "__main__":
    launch()
