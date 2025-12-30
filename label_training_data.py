#!/usr/bin/env python3
"""
MONSTER BODY PART LABELING TOOL
================================

Draw bounding boxes around body parts to create training data.

Usage:
    python label_training_data.py

Controls:
    - Click and drag to draw boxes
    - Select body part from dropdown before drawing
    - Press 'S' to save
    - Press 'N' for next image
    - Press 'P' for previous image
    - Press 'Z' to undo last box
    - Press 'D' to delete selected box
"""

import os
import json
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import shutil

# Directories
TRAINING_DIR = Path(__file__).parent / "training_data"
IMAGES_DIR = TRAINING_DIR / "images"
LABELS_FILE = TRAINING_DIR / "labels.json"

# Body parts list
BODY_PARTS = [
    "head", "skull", "face", "eye", "eyes", "mouth", "jaw", "teeth", "fangs",
    "tongue", "nose", "snout", "ear", "ears", "horn", "horns", "antler", "antlers",
    "neck", "throat", "chest", "torso", "body", "stomach", "belly", "back", "spine",
    "shoulder", "shoulders", "arm", "arms", "forearm", "elbow", "wrist",
    "hand", "hands", "finger", "fingers", "claw", "claws", "talon", "talons",
    "paw", "paws", "fist", "palm",
    "hip", "hips", "leg", "legs", "thigh", "knee", "shin", "ankle",
    "foot", "feet", "toe", "toes", "hoof", "hooves",
    "tail", "tails", "tentacle", "tentacles", "appendage", "limb",
    "wing", "wings", "feather", "feathers", "fin", "fins",
    "fur", "hair", "mane", "beard", "whiskers", "scales", "skin", "hide",
    "shell", "carapace", "exoskeleton",
    "weapon", "sword", "axe", "hammer", "staff", "spear", "blade", "dagger",
    "shield", "armor", "helmet", "gauntlet", "cape", "cloak", "robe",
    # Custom parts - add your own!
    "other"
]

# Colors for visualization
COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#FF8000", "#8000FF", "#00FF80", "#FF0080", "#80FF00", "#0080FF",
    "#FF4444", "#44FF44", "#4444FF", "#FFAA00", "#AA00FF", "#00FFAA"
]


class LabelingState:
    def __init__(self):
        self.images = []
        self.current_idx = 0
        self.labels = {}  # {image_name: [{"label": "head", "bbox": [x1,y1,x2,y2]}, ...]}
        self.current_boxes = []
        self.drawing = False
        self.start_point = None
        self.load_data()

    def load_data(self):
        """Load existing labels and find images"""
        # Create directories if needed
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        # Find all images
        self.images = sorted([
            f.name for f in IMAGES_DIR.glob("*")
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
        ])

        # Load existing labels
        if LABELS_FILE.exists():
            with open(LABELS_FILE) as f:
                data = json.load(f)
                for item in data:
                    self.labels[item["image"]] = item["boxes"]

        print(f"Found {len(self.images)} images, {len(self.labels)} labeled")

    def save_data(self):
        """Save all labels to JSON"""
        data = []
        for img_name, boxes in self.labels.items():
            if boxes:  # Only save if has labels
                data.append({
                    "image": img_name,
                    "boxes": boxes
                })

        with open(LABELS_FILE, "w") as f:
            json.dump(data, f, indent=2)

        return f"Saved {len(data)} labeled images to {LABELS_FILE}"

    def get_current_image(self):
        """Get current image with boxes drawn"""
        if not self.images:
            return None, "No images found. Add images to training_data/images/"

        img_name = self.images[self.current_idx]
        img_path = IMAGES_DIR / img_name
        img = Image.open(img_path).convert("RGB")

        # Get boxes for this image
        boxes = self.labels.get(img_name, [])

        # Draw boxes
        draw = ImageDraw.Draw(img)
        for i, box in enumerate(boxes):
            color = COLORS[i % len(COLORS)]
            x1, y1, x2, y2 = box["bbox"]
            label = box["label"]

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label
            draw.rectangle([x1, y1-20, x1+len(label)*8+10, y1], fill=color)
            draw.text((x1+5, y1-18), label, fill="white")

        status = f"Image {self.current_idx + 1}/{len(self.images)}: {img_name} | {len(boxes)} boxes"
        return np.array(img), status

    def add_box(self, label, x1, y1, x2, y2):
        """Add a bounding box to current image"""
        if not self.images:
            return

        img_name = self.images[self.current_idx]

        # Ensure coordinates are in right order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Skip tiny boxes
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            return

        if img_name not in self.labels:
            self.labels[img_name] = []

        self.labels[img_name].append({
            "label": label,
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })

    def undo_last_box(self):
        """Remove last box from current image"""
        if not self.images:
            return

        img_name = self.images[self.current_idx]
        if img_name in self.labels and self.labels[img_name]:
            self.labels[img_name].pop()

    def clear_boxes(self):
        """Clear all boxes from current image"""
        if not self.images:
            return

        img_name = self.images[self.current_idx]
        self.labels[img_name] = []

    def next_image(self):
        """Go to next image"""
        if self.images:
            self.current_idx = (self.current_idx + 1) % len(self.images)

    def prev_image(self):
        """Go to previous image"""
        if self.images:
            self.current_idx = (self.current_idx - 1) % len(self.images)

    def go_to_image(self, idx):
        """Go to specific image index"""
        if self.images and 0 <= idx < len(self.images):
            self.current_idx = idx


# Global state
STATE = LabelingState()


def refresh_image():
    """Refresh the displayed image"""
    return STATE.get_current_image()


def add_box_from_coords(label, x1, y1, x2, y2):
    """Add box from coordinate inputs"""
    STATE.add_box(label, x1, y1, x2, y2)
    return refresh_image()


def handle_click(image, label, evt: gr.SelectData):
    """Handle click on image for box drawing"""
    global STATE

    if not hasattr(STATE, 'click_start'):
        STATE.click_start = None

    if STATE.click_start is None:
        # First click - start box
        STATE.click_start = (evt.index[0], evt.index[1])
        return image, f"Click second corner to complete box for '{label}'"
    else:
        # Second click - complete box
        x1, y1 = STATE.click_start
        x2, y2 = evt.index[0], evt.index[1]
        STATE.click_start = None

        STATE.add_box(label, x1, y1, x2, y2)
        img, status = refresh_image()
        return img, status + " | Box added!"


def undo_box():
    """Undo last box"""
    STATE.undo_last_box()
    return refresh_image()


def clear_all():
    """Clear all boxes"""
    STATE.clear_boxes()
    return refresh_image()


def next_img():
    """Next image"""
    STATE.next_image()
    return refresh_image()


def prev_img():
    """Previous image"""
    STATE.prev_image()
    return refresh_image()


def save_labels():
    """Save labels to file"""
    msg = STATE.save_data()
    img, status = refresh_image()
    return img, msg


def add_images_from_folder(folder_path):
    """Copy images from a folder to training directory"""
    if not folder_path or not os.path.isdir(folder_path):
        return refresh_image()[0], "Invalid folder path"

    count = 0
    for f in Path(folder_path).glob("*"):
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            dest = IMAGES_DIR / f.name
            if not dest.exists():
                shutil.copy(f, dest)
                count += 1

    STATE.load_data()
    img, status = refresh_image()
    return img, f"Added {count} images. Total: {len(STATE.images)} images"


def get_stats():
    """Get labeling statistics"""
    total_images = len(STATE.images)
    labeled = len([i for i in STATE.images if STATE.labels.get(i)])
    total_boxes = sum(len(boxes) for boxes in STATE.labels.values())

    # Count by label type
    label_counts = {}
    for boxes in STATE.labels.values():
        for box in boxes:
            label = box["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

    stats = f"""
    ═══════════════════════════════════════
    TRAINING DATA STATISTICS
    ═══════════════════════════════════════
    Total Images: {total_images}
    Labeled Images: {labeled}
    Total Boxes: {total_boxes}
    ───────────────────────────────────────
    """

    if label_counts:
        stats += "    Labels:\n"
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            stats += f"      {label}: {count}\n"

    return stats


# Build UI
with gr.Blocks(title="Monster Body Part Labeler") as demo:
    gr.Markdown("""
    # 🎯 Monster Body Part Labeler

    **Create training data for Florence-2 fine-tuning**

    1. Add monster images to `training_data/images/` folder
    2. Select a body part from the dropdown
    3. Click two corners on the image to draw a box
    4. Save regularly!
    """)

    with gr.Row():
        with gr.Column(scale=3):
            image = gr.Image(
                label="Image (click to draw boxes)",
                interactive=False,
                height=600
            )
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### Body Part")
            label_dropdown = gr.Dropdown(
                choices=BODY_PARTS,
                value="head",
                label="Select part to label"
            )

            gr.Markdown("### Navigation")
            with gr.Row():
                prev_btn = gr.Button("◀ Prev")
                next_btn = gr.Button("Next ▶")

            with gr.Row():
                undo_btn = gr.Button("↩ Undo Last")
                clear_btn = gr.Button("🗑 Clear All")

            save_btn = gr.Button("💾 SAVE LABELS", variant="primary", size="lg")

            gr.Markdown("### Quick Add Box")
            with gr.Row():
                x1 = gr.Number(label="X1", value=0, scale=1)
                y1 = gr.Number(label="Y1", value=0, scale=1)
            with gr.Row():
                x2 = gr.Number(label="X2", value=100, scale=1)
                y2 = gr.Number(label="Y2", value=100, scale=1)
            add_box_btn = gr.Button("Add Box")

            gr.Markdown("### Import Images")
            folder_input = gr.Textbox(
                label="Folder Path",
                placeholder="C:/path/to/monsters"
            )
            import_btn = gr.Button("Import from Folder")

            gr.Markdown("### Statistics")
            stats_btn = gr.Button("Show Stats")
            stats_display = gr.Textbox(label="Stats", lines=10, interactive=False)

    # Event handlers
    image.select(
        fn=handle_click,
        inputs=[image, label_dropdown],
        outputs=[image, status]
    )

    prev_btn.click(fn=prev_img, outputs=[image, status])
    next_btn.click(fn=next_img, outputs=[image, status])
    undo_btn.click(fn=undo_box, outputs=[image, status])
    clear_btn.click(fn=clear_all, outputs=[image, status])
    save_btn.click(fn=save_labels, outputs=[image, status])

    add_box_btn.click(
        fn=lambda l, a, b, c, d: (STATE.add_box(l, a, b, c, d), refresh_image())[1],
        inputs=[label_dropdown, x1, y1, x2, y2],
        outputs=[image, status]
    )

    import_btn.click(
        fn=add_images_from_folder,
        inputs=[folder_input],
        outputs=[image, status]
    )

    stats_btn.click(fn=get_stats, outputs=[stats_display])

    # Load initial image
    demo.load(fn=refresh_image, outputs=[image, status])


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          MONSTER BODY PART LABELING TOOL                     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  1. Put monster images in: training_data/images/             ║
    ║  2. Select body part from dropdown                           ║
    ║  3. Click two corners on image to draw box                   ║
    ║  4. Click SAVE regularly!                                    ║
    ║                                                              ║
    ║  When done, run: python train_florence2.py                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    demo.launch(server_name="127.0.0.1")
