#!/usr/bin/env python3
"""VEILBREAKERS Monster Rigger - Launcher"""
import sys

print("=" * 50)
print("    VEILBREAKERS MONSTER RIGGER v3.0")
print("=" * 50)
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
    except:
        print(f"  [--] {name} (optional)")

print()
print("=" * 50)
print("Launching UI...")
print("=" * 50)
print()

from working_ui import demo, init, preload_models
init()
preload_models()
demo.launch(server_name="127.0.0.1")
