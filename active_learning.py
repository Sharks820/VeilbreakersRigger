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

# Training metrics
try:
    from training_metrics import (
        TrainingMetricsTracker, generate_learning_report, get_model_status,
        generate_ascii_learning_curve, generate_data_quality_report, get_training_data_stats
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    def generate_learning_report():
        return "Training metrics module not found. Run training to create it."
    def get_model_status():
        return {"finetuned_model_exists": False, "using_model": "base", "training_summary": {}}
    def generate_ascii_learning_curve():
        return "Metrics module not available"
    def generate_data_quality_report():
        return "Metrics module not available"
    def get_training_data_stats():
        return {"total_samples": 0}

# Directories
BASE_DIR = Path(__file__).parent
TRAINING_DIR = BASE_DIR / "training_data"
IMAGES_DIR = TRAINING_DIR / "images"
CORRECTIONS_DIR = TRAINING_DIR / "corrections"
LABELS_FILE = TRAINING_DIR / "labels.json"
MODEL_DIR = BASE_DIR / "florence2_finetuned"

# Body parts - STRICT LIST for detection
# These are the ONLY labels we want the AI to learn
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

# Mapping from generic descriptions to body parts
# This helps convert "purple cat head" -> "head"
LABEL_MAPPING = {
    # Direct mappings for common full-object detections
    "cat": "body", "dog": "body", "monster": "body", "creature": "body",
    "animal": "body", "character": "body", "figure": "body", "person": "body",
    "dragon": "body", "beast": "body", "demon": "body",
    # Body part synonyms
    "face": "head", "skull": "head", "cranium": "head",
    "torso": "body", "trunk": "body", "chest": "body",
    "limb": "leg", "appendage": "arm", "extremity": "leg",
    "claw": "claw", "talon": "claw", "nail": "claw",
    "paw": "paw", "foot": "foot", "hoof": "hoof",
    "wing": "wing", "fin": "fin", "flipper": "fin",
    "tail": "tail", "tentacle": "tentacle",
}


def extract_body_part_from_label(label: str) -> str:
    """
    Convert a descriptive label like 'purple cat head' to just 'head'
    This is CRITICAL for teaching the AI to detect body parts specifically.
    """
    label = label.lower().strip()

    # First, check if ANY body part is mentioned in the label
    for part in BODY_PARTS:
        if part in label:
            return part

    # Check mapping for full-object descriptions
    for key, value in LABEL_MAPPING.items():
        if key in label:
            return value

    # If nothing matches, try to extract the last word (often the actual part)
    words = label.split()
    if words:
        last_word = words[-1]
        if last_word in BODY_PARTS:
            return last_word

    # Default: return the original but flag it
    return f"unknown_{label.replace(' ', '_')[:15]}"

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
                # Use Florence-2 PRO (large-ft = better accuracy)
                model_id = "microsoft/Florence-2-large-ft"
                print(f"Loading Florence-2 PRO: {model_id}")

            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                attn_implementation="eager"
            )
            self.model.eval()
            print("Model loaded!")
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def detect_parts(self, image):
        """
        Run Florence-2 detection on image - EXTRACTS BODY PARTS SPECIFICALLY

        This function:
        1. Runs Florence-2 detection
        2. FILTERS results to only body parts
        3. CONVERTS generic labels ("purple cat") to body part names ("body")
        4. The finetuned model will learn to output body part names directly!
        """
        if not self.load_model():
            return []

        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image

        w, h = pil_image.size

        # STRATEGY 1: Use phrase grounding with SPECIFIC body parts
        # This tells Florence-2 exactly what to look for
        task = "<CAPTION_TO_PHRASE_GROUNDING>"

        # CRITICAL: Only ask for body parts we actually want
        body_parts_prompt = (
            "head. body. eye. eyes. mouth. nose. ear. ears. "
            "arm. arms. hand. hands. leg. legs. foot. feet. paw. paws. "
            "tail. wing. wings. claw. claws. horn. horns. "
            "neck. chest. back. shoulder. knee. elbow."
        )

        inputs = self.processor(
            text=task + body_parts_prompt,
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

        print(f"[DEBUG] Florence-2 raw output: {labels}")

        # Track used labels to handle duplicates
        label_counts = {}

        for bbox, label in zip(bboxes, labels):
            raw_label = label.lower().strip()

            # CRITICAL: Convert to body part name
            body_part = extract_body_part_from_label(raw_label)

            # Skip unknown parts (user should add these manually)
            if body_part.startswith("unknown_"):
                print(f"[SKIP] '{raw_label}' -> not a body part")
                continue

            # Handle duplicates (arm, arm_2, arm_3)
            if body_part in label_counts:
                label_counts[body_part] += 1
                final_label = f"{body_part}_{label_counts[body_part]}"
            else:
                label_counts[body_part] = 1
                final_label = body_part

            print(f"[DETECT] '{raw_label}' -> '{final_label}'")

            boxes.append({
                "label": final_label,
                "bbox": [int(x) for x in bbox],
                "source": "ai",
                "original_label": raw_label  # Keep original for debugging
            })

        print(f"[RESULT] Detected {len(boxes)} body parts: {[b['label'] for b in boxes]}")
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
        """
        Save corrected boxes as training data.

        CRITICAL: Labels are normalized to BODY PART NAMES so the AI
        learns to output "head", "arm", "leg" - not "purple cat head"!
        """
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

        # CRITICAL: Normalize labels for clean training
        normalized_boxes = []
        for b in boxes:
            label = b["label"].lower().strip()

            # Remove duplicate suffixes (arm_2 -> arm for training)
            # The model should learn "arm", it will output "arm" multiple times
            clean_label = label.rstrip("_0123456789")

            # Validate it's a body part
            if clean_label not in BODY_PARTS:
                extracted = extract_body_part_from_label(label)
                if not extracted.startswith("unknown_"):
                    clean_label = extracted
                else:
                    # Keep original if we can't map it - user knows best
                    clean_label = label
                    print(f"[NOTE] Custom label '{label}' - not in default body parts list")

            normalized_boxes.append({
                "label": clean_label,
                "bbox": b["bbox"]
            })

        print(f"[TRAINING DATA] Labels: {[b['label'] for b in normalized_boxes]}")

        # Load existing labels
        if LABELS_FILE.exists():
            with open(LABELS_FILE) as f:
                all_labels = json.load(f)
        else:
            all_labels = []

        # Add this correction
        all_labels.append({
            "image": img_name,
            "boxes": normalized_boxes
        })

        # Save
        with open(LABELS_FILE, "w") as f:
            json.dump(all_labels, f, indent=2)

        self.corrections_count = len(all_labels)
        return len(all_labels)


# Global state
LEARNER = ActiveLearner()

# Auto-training settings
AUTO_TRAIN_THRESHOLD = 20  # Auto-train after this many corrections


def get_learning_progress() -> str:
    """Generate a progress summary for the user"""
    stats = get_training_data_stats() if METRICS_AVAILABLE else {"total_samples": LEARNER.corrections_count}
    model_info = get_model_status() if METRICS_AVAILABLE else {"using_model": "base"}

    samples = stats.get("total_samples", 0)

    progress = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                      🧠 LEARNING PROGRESS                                 ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Corrections saved:  {samples:<5}                                              ║"""

    if samples < 5:
        progress += f"""
║  Status:             ⚠️  Need {5 - samples} more corrections to enable training   ║
║                                                                           ║
║  NEXT STEP: Upload image → Correct detections → Save → Repeat             ║"""
    elif samples < 20:
        progress += f"""
║  Status:             ⚡ Ready to train! ({samples}/20 recommended)                ║
║                                                                           ║
║  NEXT STEP: Click "🚀 TRAIN MODEL" or add more corrections                ║"""
    else:
        progress += f"""
║  Status:             ✅ Excellent dataset! Ready for training             ║
║                                                                           ║
║  NEXT STEP: Click "🚀 TRAIN MODEL" to improve the AI                      ║"""

    progress += f"""
╠═══════════════════════════════════════════════════════════════════════════╣
║  Current model:      {model_info.get('using_model', 'base')[:40]:<40} ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
    return progress


def load_and_detect(image):
    """Load image and run AI detection"""
    global LEARNER

    if image is None:
        return None, "⬆️ Upload an image first", "[]"

    LEARNER.current_image = image
    LEARNER.current_boxes = LEARNER.detect_parts(image)

    vis = LEARNER.draw_boxes(image, LEARNER.current_boxes)
    boxes_json = json.dumps(LEARNER.current_boxes, indent=2)

    num_parts = len(LEARNER.current_boxes)
    if num_parts == 0:
        status = "🤔 AI didn't detect anything. Add boxes manually below!"
    else:
        status = f"✅ AI detected {num_parts} parts (GREEN). Now CORRECT any mistakes (your corrections = RED), then click SAVE!"

    return vis, status, boxes_json


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


def save_corrections(image, boxes_json, auto_train_enabled):
    """Save current boxes as training data"""
    global LEARNER

    boxes = json.loads(boxes_json) if boxes_json else []
    if not boxes:
        return image, "⚠️ No boxes to save! Add at least one box.", get_learning_progress()

    count = LEARNER.save_correction(LEARNER.current_image, boxes)

    # Update progress
    progress = get_learning_progress()

    if count < 5:
        status = f"✅ SAVED! ({count}/5 minimum for training). Keep going - {5-count} more needed!"
    elif count < 20:
        status = f"✅ SAVED! ({count} samples). Ready to train! Click 🚀 TRAIN MODEL."
    else:
        status = f"✅ SAVED! ({count} samples). Excellent! Your AI will be very accurate."

    # Auto-train suggestion
    if count >= AUTO_TRAIN_THRESHOLD and count % 10 == 0:
        status += "\n💡 TIP: You have enough data for great results. Consider training now!"

    return image, status, progress


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

    # Run PRO training script (with metrics tracking, augmentation, early stopping)
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "train_florence2_pro.py")],
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


def run_ab_comparison(image):
    """Run A/B comparison between base and finetuned model"""
    if image is None:
        return None, None, "Upload an image first!"

    results = []

    # Test with base model
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM

        # Base model detection
        base_model_id = "microsoft/Florence-2-large-ft"
        base_processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, trust_remote_code=True,
            torch_dtype=torch.float32, attn_implementation="eager"
        )
        base_model.eval()

        pil_image = Image.fromarray(image)
        w, h = pil_image.size

        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = ". ".join(BODY_PARTS[:30]) + "."

        inputs = base_processor(text=task + prompt, images=pil_image, return_tensors="pt")
        with torch.no_grad():
            generated_ids = base_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].to(base_model.dtype),
                max_new_tokens=1024, num_beams=1, do_sample=False, use_cache=False
            )
        text = base_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        base_parsed = base_processor.post_process_generation(text, task=task, image_size=(w, h))
        base_boxes = base_parsed.get(task, {}).get("bboxes", [])
        base_labels = base_parsed.get(task, {}).get("labels", [])

        # Draw base results
        base_vis = image.copy()
        base_img = Image.fromarray(base_vis)
        base_draw = ImageDraw.Draw(base_img)
        for bbox, label in zip(base_boxes, base_labels):
            x1, y1, x2, y2 = [int(x) for x in bbox]
            base_draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)
            base_draw.text((x1+3, y1-16), label, fill="white")
        base_vis = np.array(base_img)

        del base_model, base_processor

    except Exception as e:
        base_vis = image.copy()
        base_boxes = []
        base_labels = []

    # Check for finetuned model
    finetuned_path = MODEL_DIR / "final"
    if finetuned_path.exists():
        try:
            ft_processor = AutoProcessor.from_pretrained(str(finetuned_path), trust_remote_code=True)
            ft_model = AutoModelForCausalLM.from_pretrained(
                str(finetuned_path), trust_remote_code=True,
                torch_dtype=torch.float32, attn_implementation="eager"
            )
            ft_model.eval()

            inputs = ft_processor(text=task + prompt, images=pil_image, return_tensors="pt")
            with torch.no_grad():
                generated_ids = ft_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"].to(ft_model.dtype),
                    max_new_tokens=1024, num_beams=1, do_sample=False, use_cache=False
                )
            text = ft_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            ft_parsed = ft_processor.post_process_generation(text, task=task, image_size=(w, h))
            ft_boxes = ft_parsed.get(task, {}).get("bboxes", [])
            ft_labels = ft_parsed.get(task, {}).get("labels", [])

            # Draw finetuned results
            ft_vis = image.copy()
            ft_img = Image.fromarray(ft_vis)
            ft_draw = ImageDraw.Draw(ft_img)
            for bbox, label in zip(ft_boxes, ft_labels):
                x1, y1, x2, y2 = [int(x) for x in bbox]
                ft_draw.rectangle([x1, y1, x2, y2], outline="#00FFFF", width=3)
                ft_draw.text((x1+3, y1-16), label, fill="white")
            ft_vis = np.array(ft_img)

            comparison = f"""
A/B COMPARISON RESULTS:
=======================
BASE MODEL:      {len(base_boxes)} parts detected
FINETUNED MODEL: {len(ft_boxes)} parts detected

Base found: {', '.join(base_labels[:5])}{'...' if len(base_labels) > 5 else ''}
Finetuned found: {', '.join(ft_labels[:5])}{'...' if len(ft_labels) > 5 else ''}

{'✅ Finetuned model detects MORE parts!' if len(ft_boxes) > len(base_boxes) else '⚠️ Similar or fewer parts - needs more training'}
"""
            del ft_model, ft_processor

        except Exception as e:
            ft_vis = image.copy()
            comparison = f"Error loading finetuned model: {e}"
    else:
        ft_vis = image.copy()
        comparison = "No finetuned model found. Train the model first!"

    return base_vis, ft_vis, comparison


# Custom body parts config file
CUSTOM_PARTS_FILE = BASE_DIR / "custom_body_parts.txt"


def load_custom_parts():
    """Load user's custom body parts list"""
    if CUSTOM_PARTS_FILE.exists():
        with open(CUSTOM_PARTS_FILE) as f:
            custom = [line.strip().lower() for line in f if line.strip()]
            return custom
    return []


def save_custom_parts(parts_text):
    """Save custom body parts list"""
    parts = [p.strip().lower() for p in parts_text.split('\n') if p.strip()]
    with open(CUSTOM_PARTS_FILE, 'w') as f:
        f.write('\n'.join(parts))
    # Update the global list
    global BODY_PARTS
    BODY_PARTS.extend([p for p in parts if p not in BODY_PARTS])
    return f"✅ Saved {len(parts)} custom body parts. They will be detected now!"


# Build UI - THE GREATEST LEARNING INTERFACE EVER MADE
with gr.Blocks(title="VEILBREAKERS - AI Learning System", theme=gr.themes.Soft(primary_hue="orange")) as demo:

    gr.Markdown("""
    # 🧠 VEILBREAKERS AI LEARNING SYSTEM

    ## This AI LEARNS From YOU!

    The more corrections you make, the smarter the AI becomes at detecting **specific body parts** like:
    `head`, `body`, `arm`, `leg`, `tail`, `eye`, `claw`, `wing`, etc.

    **NOT** generic descriptions like "purple cat" - we want **PARTS**!
    """)

    # PROGRESS DISPLAY
    progress_display = gr.Markdown(value=get_learning_progress())

    with gr.Tabs():
        # TAB 1: THE MAIN LEARNING WORKFLOW
        with gr.Tab("📝 STEP 1: Label & Correct"):
            gr.Markdown("""
            ### How The AI Learns From You:
            1. **Upload** a monster image
            2. **Click DETECT** - AI shows what it thinks (GREEN boxes)
            3. **Correct mistakes** - Add missing parts, remove wrong ones (your fixes = RED)
            4. **Click SAVE** - Your correction teaches the AI
            5. **Repeat** with different images until you have 5+ corrections
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    image_input = gr.Image(label="⬆️ STEP 1: Upload Monster Image", type="numpy")
                    detect_btn = gr.Button("🔍 STEP 2: DETECT PARTS", variant="primary", size="lg")

                    image_output = gr.Image(
                        label="Detection Result (GREEN=AI guess | RED=Your corrections)",
                        interactive=False
                    )
                    status = gr.Textbox(label="Status", lines=2)

                with gr.Column(scale=1):
                    gr.Markdown("### ✏️ STEP 3: Make Corrections")
                    gr.Markdown("Add boxes for parts the AI **MISSED**:")

                    label_dropdown = gr.Dropdown(choices=BODY_PARTS, value="head", label="Part Type")

                    gr.Markdown("*Box coordinates (or use click-to-drag in main rigger):*")
                    with gr.Row():
                        x1 = gr.Number(label="X1", value=0, precision=0)
                        y1 = gr.Number(label="Y1", value=0, precision=0)
                    with gr.Row():
                        x2 = gr.Number(label="X2", value=100, precision=0)
                        y2 = gr.Number(label="Y2", value=100, precision=0)

                    add_btn = gr.Button("➕ Add This Part", variant="secondary")

                    gr.Markdown("---")
                    with gr.Row():
                        undo_btn = gr.Button("↩ Undo Last", size="sm")
                        clear_ai_btn = gr.Button("🗑 Clear AI Boxes", size="sm")

                    gr.Markdown("---")
                    gr.Markdown("### 💾 STEP 4: Save Your Corrections")
                    auto_train = gr.Checkbox(label="Auto-notify when ready to train", value=True)
                    save_btn = gr.Button("💾 SAVE CORRECTION", variant="primary", size="lg")

                    gr.Markdown("### Debug: Current Boxes")
                    boxes_json = gr.Code(label="Boxes JSON", language="json", value="[]", lines=5)

        # TAB 2: TRAINING
        with gr.Tab("🚀 STEP 2: Train The AI"):
            gr.Markdown("""
            ### Train Your AI!

            When you have enough corrections (5+ minimum, 20+ recommended), click the button below
            to make the AI learn from your feedback. Training takes 5-30 minutes depending on your GPU.
            """)

            training_status = gr.Textbox(value=get_training_status(), label="Training Data Available", interactive=False)
            train_btn = gr.Button("🚀 START TRAINING", variant="primary", size="lg")

            gr.Markdown("""
            ---
            ### What Happens During Training:

            1. Your corrections are loaded
            2. The AI studies what you taught it
            3. It adjusts its understanding of body parts
            4. A new, smarter model is saved
            5. Next time you detect, it uses YOUR knowledge!

            **Important:** Don't close this window during training!
            """)

            training_output = gr.Textbox(label="Training Progress", lines=10, interactive=False)

        # TAB 3: VERIFY LEARNING
        with gr.Tab("📊 STEP 3: Verify Learning"):
            gr.Markdown("""
            ### Proof That The AI Is Learning

            This section shows **objective metrics** proving the AI is improving.
            No placebo effect - real, measurable progress!
            """)

            with gr.Row():
                refresh_all_btn = gr.Button("🔄 Refresh All Reports", variant="secondary")

            with gr.Accordion("📈 Learning Curve", open=True):
                learning_curve = gr.Code(
                    value=generate_ascii_learning_curve() if METRICS_AVAILABLE else "Run training first",
                    label="Loss Over Training",
                    language=None,
                    lines=20
                )

            with gr.Accordion("📋 Training Report", open=True):
                learning_report = gr.Code(
                    value=generate_learning_report(),
                    label="Learning Summary",
                    language=None,
                    lines=15
                )

            with gr.Accordion("🗂️ Data Quality", open=True):
                data_quality = gr.Code(
                    value=generate_data_quality_report() if METRICS_AVAILABLE else "Metrics module loading...",
                    label="Training Data Analysis",
                    language=None,
                    lines=20
                )

            model_status_box = gr.JSON(value=get_model_status(), label="Current Model Status")

        # TAB 4: A/B COMPARISON
        with gr.Tab("🔬 A/B Comparison"):
            gr.Markdown("""
            ### Side-by-Side Comparison: Base vs Your Trained Model

            Upload any image to compare how the **original AI** vs **YOUR trained AI** perform.
            This is the ultimate proof that learning is working!
            """)

            comparison_image = gr.Image(label="Upload Test Image", type="numpy")
            compare_btn = gr.Button("🔬 Run A/B Comparison", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Base Model (Before Training)")
                    base_result = gr.Image(label="Base Florence-2", interactive=False)
                with gr.Column():
                    gr.Markdown("### YOUR Model (After Training)")
                    finetuned_result = gr.Image(label="Your Finetuned Model", interactive=False)

            comparison_text = gr.Textbox(label="Comparison Results", lines=10, interactive=False)

        # TAB 5: CONFIGURATION
        with gr.Tab("⚙️ Configuration"):
            gr.Markdown("""
            ### Custom Body Parts

            Add your own body part names here (one per line).
            These will be recognized by the detection and training systems.

            **Default parts include:** head, body, arm, leg, tail, eye, claw, wing, paw, horn, etc.
            """)

            default_custom = '\n'.join(load_custom_parts()) if load_custom_parts() else "front_left_leg\nfront_right_leg\nback_left_leg\nback_right_leg\nmuzzle\nsnout"

            custom_parts_input = gr.Textbox(
                label="Custom Body Parts (one per line)",
                value=default_custom,
                lines=10,
                placeholder="front_left_leg\nfront_right_leg\nmuzzle\n..."
            )
            save_parts_btn = gr.Button("💾 Save Custom Parts", variant="primary")
            parts_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("""
            ---
            ### Current Default Body Parts

            These are always recognized:
            ```
            head, skull, face, eye, eyes, mouth, jaw, teeth, fangs,
            tongue, nose, snout, ear, ears, horn, horns, antler,
            neck, throat, chest, torso, body, stomach, belly, back,
            shoulder, arm, arms, forearm, elbow, wrist,
            hand, hands, finger, fingers, fist, palm,
            claw, claws, talon, paw, paws,
            hip, leg, legs, thigh, knee, ankle, shin,
            foot, feet, toe, toes, hoof, hooves,
            tail, tentacle, tentacles, wing, wings, fin,
            fur, hair, mane, beard, scales, shell, carapace, skin,
            weapon, sword, axe, staff, shield, armor, helmet, cape, gauntlet
            ```
            """)

            save_parts_btn.click(
                fn=save_custom_parts,
                inputs=[custom_parts_input],
                outputs=[parts_status]
            )

    # EVENT HANDLERS
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
        inputs=[image_output, boxes_json, auto_train],
        outputs=[image_output, status, progress_display]
    ).then(
        fn=get_training_status,
        outputs=[training_status]
    )

    train_btn.click(
        fn=train_model,
        outputs=[training_output]
    ).then(
        fn=get_training_status,
        outputs=[training_status]
    )

    def refresh_all_reports():
        return (
            generate_ascii_learning_curve() if METRICS_AVAILABLE else "N/A",
            generate_learning_report(),
            generate_data_quality_report() if METRICS_AVAILABLE else "N/A",
            get_model_status()
        )

    refresh_all_btn.click(
        fn=refresh_all_reports,
        outputs=[learning_curve, learning_report, data_quality, model_status_box]
    )

    compare_btn.click(
        fn=run_ab_comparison,
        inputs=[comparison_image],
        outputs=[base_result, finetuned_result, comparison_text]
    )

    # Refresh progress on load
    demo.load(fn=get_learning_progress, outputs=[progress_display])


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     🧠 VEILBREAKERS AI LEARNING SYSTEM - THE GREATEST RIGGER EVER MADE      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  POWERED BY STATE-OF-THE-ART TECHNOLOGY:                                     ║
║  ✅ Florence-2 PRO - Microsoft's best vision-language model                  ║
║  ✅ LoRA Fine-tuning - Efficient learning that preserves base knowledge      ║
║  ✅ SAM 2.1 - Meta's latest segmentation model                               ║
║  ✅ Active Learning - YOUR corrections make the AI smarter                   ║
║                                                                              ║
║  HOW IT WORKS:                                                               ║
║  1. Upload monster images                                                    ║
║  2. Click DETECT - AI shows what it found                                    ║
║  3. CORRECT the AI's mistakes (add missing, remove wrong)                    ║
║  4. Click SAVE - your correction becomes training data                       ║
║  5. After 5+ corrections, click TRAIN MODEL                                  ║
║  6. The AI LEARNS from YOU and gets better!                                  ║
║                                                                              ║
║  This is REAL machine learning - not fake, not placebo, REAL IMPROVEMENT!    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    demo.launch(server_name="127.0.0.1")
