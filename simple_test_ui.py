"""Simple test UI to verify Gradio works"""
import gradio as gr
import numpy as np
from PIL import Image

def process_image(image):
    """Simple image processing test"""
    if image is None:
        return None, "No image loaded"

    # Just return the image back with some info
    h, w = image.shape[:2]
    return image, f"Image loaded: {w}x{h}"

def on_click(image, evt: gr.SelectData):
    """Handle click"""
    if image is None:
        return None, "No image"
    x, y = evt.index
    return image, f"Clicked at ({x}, {y})"

with gr.Blocks(title="Simple Test") as demo:
    gr.Markdown("# Simple Gradio Test")

    with gr.Row():
        input_img = gr.Image(label="Upload Image", type="numpy")
        output_img = gr.Image(label="Output", type="numpy", interactive=False)

    status = gr.Textbox(label="Status", value="Upload an image to test")

    input_img.upload(process_image, inputs=[input_img], outputs=[output_img, status])
    output_img.select(on_click, inputs=[output_img], outputs=[output_img, status])

if __name__ == "__main__":
    print("Starting simple test UI...")
    demo.launch(server_name="127.0.0.1")
