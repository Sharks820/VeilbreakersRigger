#!/usr/bin/env python3
"""
ACTIVE LEARNING SYSTEM FOR FLORENCE-2
======================================

Workflow:
1. Load image -> AI detects parts
2. You CORRECT mistakes (add missing, remove wrong, fix boxes)
3. Corrections saved as training data automatically
4. Click "TRAIN" to improve model with your corrections
5. Repeat - model gets better each time!

This is how AI LEARNS from YOUR expertise!
"""

import os
import json
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from datetime import datetime
import shutil
import torch

# Directories
BASE_DIR = Path(__file__).parent
TRAINING_DIR = BASE_DIR / "training_data"
IMAGES_DIR = TRAINING_DIR / "images"
CORRECTIONS_DIR = TRAINING_DIR / "corrections"
LABELS_FILE = TRAINING_DIR / "labels.json"
MODEL_DIR = BASE_DIR / "florence2_finetuned"

# Body parts
BODY_PARTS = [
    # Head/face
    "head", "skull", "face", "eye", "eyes", "mouth", "jaw", "teeth", "fangs",
    "tongue", "nose", "snout", "ear", "ears", "horn", "horns", "antler",
    # Upper body
    "neck", "throat", "chest", "torso", "body", "stomach", "belly", "back",
    # Arms & hands (ALL variations)
    "shoulder", "arm", "arms", "forearm", "elbow", "wrist",
    "hand", "hands", "finger", "fingers", "fist", "palm",
    "claw", "claws", "talon", "paw", "paws",
    # Legs & feet (ALL variations)
    "hip", "leg", "legs", "thigh", "knee", "ankle", "shin",
    "foot", "feet", "toe", "toes", "hoof", "hooves",
    # Extras
    "tail", "tentacle", "tentacles", "wing", "wings", "fin",
    # Appearance
    "fur", "hair", "mane", "beard", "scales", "shell", "carapace", "skin",
    # Equipment
    "weapon", "sword", "axe", "staff", "shield", "armor", "helmet", "cape", "gauntlet"
]

COLORS = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
          "#FF8000", "#8000FF", "#00FF80", "#FF0080", "#80FF00", "#0080FF"]


class ActiveLearner:
    def __init__(self):
        self.current_image = None
        self.current_image_path = None
        self.current_boxes = []  # {"label": str, "bbox": [x1,y1,x2,y2], "source": "ai"|"user"}
        self.model = None
        self.processor = None
        self.corrections_count = 0
        self.setup_dirs()
        self.load_corrections_count()

    def setup_dirs(self):
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(exist_ok=True)
        CORRECTIONS_DIR.mkdir(exist_ok=True)

    def load_corrections_count(self):
        if LABELS_FILE.exists():
            with open(LABELS_FILE) as f:
                data = json.load(f)
                self.corrections_count = len(data)

    def load_model(self):
        """Load Florence-2 model (finetuned if available, else base)"""
        if self.model is not None:
            return True

        try:
            from transformers import AutoProcessor, AutoModelForCausalLM

            # Use finetuned model if exists, else base
            if (MODEL_DIR / "final").exists():
                model_id = str(MODEL_DIR / "final")
                print(f"Loading FINE-TUNED model from {model_id}")
            else:
                model_id = "microsoft/Florence-2-large"
                print(f"Loading base model: {model_id}")

            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                dtype=torch.float32,
                attn_implementation="eager"
            )
            self.model.eval()
            print("Model loaded!")
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def detect_parts(self, image):
        """Run Florence-2 detection on image"""
        if not self.load_model():
            return []

        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image

        w, h = pil_image.size

        # Use phrase grounding with body parts prompt
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = ". ".join(BODY_PARTS[:30]) + "."  # Top 30 parts

        inputs = self.processor(
            text=task + prompt,
            images=pil_image,
            return_tensors="pt"
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].to(self.model.dtype),
                max_new_tokens=1024,
                num_beams=1,
                do_sample=False,
                use_cache=False
            )

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task=task, image_size=(w, h))

        boxes = []
        bboxes = parsed.get(task, {}).get("bboxes", [])
        labels = parsed.get(task, {}).get("labels", [])

        for bbox, label in zip(bboxes, labels):
            boxes.append({
                "label": label.lower().strip(),
                "bbox": [int(x) for x in bbox],
                "source": "ai"
            })

        return boxes

    def draw_boxes(self, image, boxes):
        """Draw boxes on image"""
        if image is None:
            return None

        img = Image.fromarray(image) if isinstance(image, np.ndarray) else image.copy()
        draw = ImageDraw.Draw(img)

        for i, box in enumerate(boxes):
            color = "#00FF00" if box["source"] == "ai" else "#FF0000"
            x1, y1, x2, y2 = box["bbox"]
            label = box["label"]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.rectangle([x1, y1-18, x1+len(label)*7+6, y1], fill=color)
            draw.text((x1+3, y1-16), label, fill="white")

        return np.array(img)

    def save_correction(self, image, boxes):
        """Save corrected boxes as training data"""
        if image is None or not boxes:
            return 0

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"correction_{timestamp}.png"
        img_path = IMAGES_DIR / img_name

        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(img_path)
        else:
            image.save(img_path)

        # Load existing labels
        if LABELS_FILE.exists():
            with open(LABELS_FILE) as f:
                all_labels = json.load(f)
        else:
            all_labels = []

        # Add this correction
        all_labels.append({
            "image": img_name,
            "boxes": [{"label": b["label"], "bbox": b["bbox"]} for b in boxes]
        })

        # Save
        with open(LABELS_FILE, "w") as f:
            json.dump(all_labels, f, indent=2)

        self.corrections_count = len(all_labels)
        return len(all_labels)


# Global state
LEARNER = ActiveLearner()


def load_and_detect(image):
    """Load image and run AI detection"""
    global LEARNER

    if image is None:
        return None, "Upload an image first", "[]"

    LEARNER.current_image = image
    LEARNER.current_boxes = LEARNER.detect_parts(image)

    vis = LEARNER.draw_boxes(image, LEARNER.current_boxes)
    boxes_json = json.dumps(LEARNER.current_boxes, indent=2)

    return vis, f"AI detected {len(LEARNER.current_boxes)} parts (green=AI). Correct them below!", boxes_json


def add_box(image, label, x1, y1, x2, y2, boxes_json):
    """Add a user box"""
    global LEARNER

    boxes = json.loads(boxes_json) if boxes_json else []
    boxes.append({
        "label": label,
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "source": "user"
    })
    LEARNER.current_boxes = boxes

    vis = LEARNER.draw_boxes(LEARNER.current_image, boxes)
    return vis, f"Added '{label}'. Total: {len(boxes)} parts", json.dumps(boxes, indent=2)


def remove_last_box(image, boxes_json):
    """Remove last box"""
    global LEARNER

    boxes = json.loads(boxes_json) if boxes_json else []
    if boxes:
        removed = boxes.pop()
        LEARNER.current_boxes = boxes
        vis = LEARNER.draw_boxes(LEARNER.current_image, boxes)
        return vis, f"Removed '{removed['label']}'. Total: {len(boxes)} parts", json.dumps(boxes, indent=2)
    return image, "No boxes to remove", boxes_json


def clear_ai_boxes(image, boxes_json):
    """Remove all AI boxes, keep user boxes"""
    global LEARNER

    boxes = json.loads(boxes_json) if boxes_json else []
    user_boxes = [b for b in boxes if b["source"] == "user"]
    LEARNER.current_boxes = user_boxes

    vis = LEARNER.draw_boxes(LEARNER.current_image, user_boxes)
    return vis, f"Cleared AI boxes. Kept {len(user_boxes)} user boxes", json.dumps(user_boxes, indent=2)


def save_corrections(image, boxes_json):
    """Save current boxes as training data"""
    global LEARNER

    boxes = json.loads(boxes_json) if boxes_json else []
    if not boxes:
        return image, "No boxes to save!"

    count = LEARNER.save_correction(LEARNER.current_image, boxes)
    return image, f"✅ SAVED! Total training samples: {count}. Train when you have 10+ samples."


def train_model():
    """Trigger training"""
    import subprocess
    import sys

    # Check if enough data
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            data = json.load(f)
        if len(data) < 5:
            return f"Need at least 5 corrections. You have {len(data)}. Keep correcting!"

    # Run training script
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "train_florence2.py")],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR)
    )

    if result.returncode == 0:
        return "✅ TRAINING COMPLETE! Restart to use improved model."
    else:
        return f"Training error: {result.stderr[:500]}"


def get_training_status():
    """Get current training data status"""
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            data = json.load(f)

        total_boxes = sum(len(item["boxes"]) for item in data)
        return f"📊 {len(data)} images labeled, {total_boxes} total boxes"
    return "📊 No training data yet. Start correcting!"


# Build UI
with gr.Blocks(title="Active Learning - Train Florence-2") as demo:
    gr.Markdown("""
    # 🧠 Active Learning: Teach Florence-2 YOUR Way

    **How it works:**
    1. Upload monster image → AI detects parts
    2. CORRECT the AI (add missing, remove wrong)
    3. Click SAVE → adds to training data
    4. After 10+ corrections → click TRAIN
    5. AI learns from YOUR corrections!

    **Green boxes** = AI detected | **Red boxes** = Your corrections
    """)

    with gr.Row():
        training_status = gr.Textbox(value=get_training_status(), label="Training Data", interactive=False)
        train_btn = gr.Button("🚀 TRAIN MODEL", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(label="Upload Monster Image", type="numpy")
            detect_btn = gr.Button("🔍 DETECT PARTS", variant="primary", size="lg")

            image_output = gr.Image(label="Detection Result (Green=AI, Red=Your corrections)")
            status = gr.Textbox(label="Status")

        with gr.Column(scale=1):
            gr.Markdown("### Add/Fix Parts")
            label_dropdown = gr.Dropdown(choices=BODY_PARTS, value="head", label="Part to add")

            gr.Markdown("*Enter box coordinates:*")
            with gr.Row():
                x1 = gr.Number(label="X1", value=0)
                y1 = gr.Number(label="Y1", value=0)
            with gr.Row():
                x2 = gr.Number(label="X2", value=100)
                y2 = gr.Number(label="Y2", value=100)

            add_btn = gr.Button("➕ Add Box", variant="secondary")

            gr.Markdown("### Quick Actions")
            with gr.Row():
                undo_btn = gr.Button("↩ Undo")
                clear_ai_btn = gr.Button("🗑 Clear AI")

            gr.Markdown("---")
            save_btn = gr.Button("💾 SAVE CORRECTION", variant="primary", size="lg")

            gr.Markdown("### Current Boxes (JSON)")
            boxes_json = gr.Code(label="Boxes", language="json", value="[]")

    # Event handlers
    detect_btn.click(
        fn=load_and_detect,
        inputs=[image_input],
        outputs=[image_output, status, boxes_json]
    )

    add_btn.click(
        fn=add_box,
        inputs=[image_output, label_dropdown, x1, y1, x2, y2, boxes_json],
        outputs=[image_output, status, boxes_json]
    )

    undo_btn.click(
        fn=remove_last_box,
        inputs=[image_output, boxes_json],
        outputs=[image_output, status, boxes_json]
    )

    clear_ai_btn.click(
        fn=clear_ai_boxes,
        inputs=[image_output, boxes_json],
        outputs=[image_output, status, boxes_json]
    )

    save_btn.click(
        fn=save_corrections,
        inputs=[image_output, boxes_json],
        outputs=[image_output, status]
    ).then(
        fn=get_training_status,
        outputs=[training_status]
    )

    train_btn.click(
        fn=train_model,
        outputs=[status]
    )


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              ACTIVE LEARNING FOR FLORENCE-2                      ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  1. Upload monster image                                         ║
    ║  2. Click DETECT - AI finds parts                                ║
    ║  3. CORRECT mistakes (add missing, remove wrong)                 ║
    ║  4. Click SAVE CORRECTION                                        ║
    ║  5. After 10+ corrections, click TRAIN MODEL                     ║
    ║                                                                  ║
    ║  The AI learns from YOUR corrections and gets better!            ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    demo.launch(server_name="127.0.0.1")
