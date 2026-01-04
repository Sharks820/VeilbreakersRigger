#!/usr/bin/env python3
"""VEILBREAKERS Monster Rigger - Launcher v5.0"""
import sys

print("=" * 60)
print("    VEILBREAKERS MONSTER RIGGER v5.0 (SIMPLIFIED)")
print("    Drop → Scan → Add Missing → Export")
print("=" * 60)
print()

# Check deps
for mod, name in [("numpy", "numpy"), ("PIL", "Pillow"), ("gradio", "gradio"), ("cv2", "opencv-python")]:
    try:
        __import__(mod)
        print(f"  [OK] {name}")
    except ImportError:
        print(f"  [X] {name} - MISSING")
        sys.exit(1)

print()

# Check AI
for mod, name in [("sam2", "SAM2"), ("groundingdino", "GroundingDINO"), ("simple_lama_inpainting", "LaMa")]:
    try:
        __import__(mod)
        print(f"  [OK] {name}")
    except ImportError:
        print(f"  [--] {name} (optional)")

print()
print("=" * 60)
print("Launching UNIFIED UI...")
print("=" * 60)
print()

# Use the unified UI (all features combined)
from veilbreakers_rigger_ui import launch_ui
launch_ui()
