#!/usr/bin/env python3
"""
FLORENCE-2 FINE-TUNING FOR MONSTER BODY PART DETECTION
=======================================================

This script fine-tunes Florence-2 to detect monster body parts better.

WORKFLOW:
1. Label images using label_training_data.py (creates training data)
2. Run this script to fine-tune the model
3. Use the fine-tuned model in the rigger

Requirements:
    pip install transformers accelerate datasets peft bitsandbytes
"""

import os
import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training config
TRAINING_DIR = Path(__file__).parent / "training_data"
OUTPUT_DIR = Path(__file__).parent / "florence2_finetuned"
# Florence-2 PRO (large-ft = pre-fine-tuned, better accuracy)
MODEL_ID = "microsoft/Florence-2-large-ft"

# Body parts we want to detect
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
    "shield", "armor", "helmet", "gauntlet", "cape", "cloak", "robe"
]


class MonsterPartsDataset(Dataset):
    """Dataset for monster body part detection training"""

    def __init__(self, data_dir: Path, processor):
        self.processor = processor
        self.samples = []

        # Load all labeled data
        labels_file = data_dir / "labels.json"
        if labels_file.exists():
            with open(labels_file) as f:
                all_labels = json.load(f)

            for item in all_labels:
                img_path = data_dir / "images" / item["image"]
                if img_path.exists():
                    self.samples.append({
                        "image_path": str(img_path),
                        "boxes": item["boxes"],  # List of {"label": "head", "bbox": [x1,y1,x2,y2]}
                    })

        logger.info(f"Loaded {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        w, h = image.size

        # Create Florence-2 style target
        # Format: "label<loc_x1><loc_y1><loc_x2><loc_y2>label2<loc_...>..."
        task = "<CAPTION_TO_PHRASE_GROUNDING>"

        # Build prompt (what we're looking for)
        labels_in_image = [box["label"] for box in sample["boxes"]]
        prompt = ". ".join(labels_in_image) + "."

        # Build target output with location tokens
        # Florence-2 uses location tokens like <loc_123> where 123 is 0-999 normalized
        target_parts = []
        for box in sample["boxes"]:
            label = box["label"]
            x1, y1, x2, y2 = box["bbox"]

            # Normalize to 0-999 range
            loc_x1 = int((x1 / w) * 999)
            loc_y1 = int((y1 / h) * 999)
            loc_x2 = int((x2 / w) * 999)
            loc_y2 = int((y2 / h) * 999)

            target_parts.append(f"{label}<loc_{loc_x1}><loc_{loc_y1}><loc_{loc_x2}><loc_{loc_y2}>")

        target = "".join(target_parts)

        # Process with Florence-2 processor
        inputs = self.processor(
            text=task + prompt,
            images=image,
            return_tensors="pt"
        )

        # Process target
        target_inputs = self.processor.tokenizer(
            target,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": target_inputs["input_ids"].squeeze(0),
        }


def setup_training():
    """Setup model and training components"""

    logger.info(f"Loading Florence-2 from {MODEL_ID}...")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for training stability
        attn_implementation="eager"
    )

    # Apply LoRA for efficient fine-tuning
    # This only trains a small portion of the model (~1% of parameters)
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def train():
    """Main training function"""

    # Check for training data
    if not TRAINING_DIR.exists():
        TRAINING_DIR.mkdir(parents=True)
        (TRAINING_DIR / "images").mkdir()

        # Create example labels file
        example_labels = [
            {
                "image": "example_monster.png",
                "boxes": [
                    {"label": "head", "bbox": [100, 50, 200, 150]},
                    {"label": "body", "bbox": [80, 150, 220, 350]},
                    {"label": "arm", "bbox": [30, 160, 80, 280]},
                    {"label": "arm", "bbox": [220, 160, 270, 280]},
                    {"label": "leg", "bbox": [90, 350, 140, 480]},
                    {"label": "leg", "bbox": [160, 350, 210, 480]},
                ]
            }
        ]

        with open(TRAINING_DIR / "labels.json", "w") as f:
            json.dump(example_labels, f, indent=2)

        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    TRAINING DATA NEEDED                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Created training directory at:                                  ║
║  {TRAINING_DIR}
║                                                                  ║
║  TO ADD TRAINING DATA:                                           ║
║                                                                  ║
║  1. Run: python label_training_data.py                           ║
║     This opens a UI to label your monster images                 ║
║                                                                  ║
║  2. Or manually:                                                 ║
║     - Put monster images in: training_data/images/               ║
║     - Edit labels.json with bounding boxes                       ║
║                                                                  ║
║  3. Then run this script again to train                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        return

    # Load labels
    labels_file = TRAINING_DIR / "labels.json"
    if not labels_file.exists():
        print("No labels.json found! Run label_training_data.py first.")
        return

    with open(labels_file) as f:
        labels = json.load(f)

    if len(labels) < 5:
        print(f"Only {len(labels)} labeled images. Need at least 5-10 for basic training.")
        print("Run: python label_training_data.py")
        return

    print(f"Found {len(labels)} labeled images. Starting training...")

    # Setup
    model, processor = setup_training()
    dataset = MonsterPartsDataset(TRAINING_DIR, processor)

    if len(dataset) == 0:
        print("No valid training samples found!")
        return

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=10,
        per_device_train_batch_size=1,  # Small batch for memory
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train!
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Samples: {len(dataset)}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60 + "\n")

    trainer.train()

    # Save final model
    trainer.save_model(str(OUTPUT_DIR / "final"))
    processor.save_pretrained(str(OUTPUT_DIR / "final"))

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {OUTPUT_DIR / 'final'}")
    print("\nTo use the fine-tuned model, update veilbreakers_rigger.py:")
    print(f'  model_id = "{OUTPUT_DIR / "final"}"')
    print("="*60)


if __name__ == "__main__":
    train()
