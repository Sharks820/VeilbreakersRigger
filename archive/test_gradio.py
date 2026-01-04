import gradio as gr

def echo(img):
    return img, f"Got image: {img.shape if img is not None else 'None'}"

demo = gr.Interface(
    fn=echo,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(), gr.Textbox()]
)

demo.launch()
