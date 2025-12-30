"""Windows-compatible test for VEILBREAKERS Rigger"""
import sys
import traceback

passed = 0
failed = 0
errors = []

def test(name, condition=True, error=None):
    global passed, failed, errors
    if condition and error is None:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        if error:
            print(f"         Error: {error}")
            errors.append((name, error))
        failed += 1

print("=" * 60)
print("  VEILBREAKERS RIGGER - WINDOWS COMPATIBILITY TEST")
print("=" * 60)

# 1. Core Imports
print("\n[1] CORE IMPORTS")
print("-" * 40)

try:
    import numpy as np
    test("numpy", True)
except Exception as e:
    test("numpy", False, str(e))

try:
    import cv2
    test("opencv (cv2)", True)
except Exception as e:
    test("opencv (cv2)", False, str(e))

try:
    from PIL import Image
    test("pillow (PIL)", True)
except Exception as e:
    test("pillow (PIL)", False, str(e))

try:
    import torch
    test(f"torch {torch.__version__}", True)
except Exception as e:
    test("torch", False, str(e))

try:
    import gradio
    test(f"gradio {gradio.__version__}", True)
except Exception as e:
    test("gradio", False, str(e))

# 2. Rigger Modules
print("\n[2] RIGGER MODULES")
print("-" * 40)

try:
    from animation_engine import AnimationEngine
    test("animation_engine.AnimationEngine", True)
except Exception as e:
    test("animation_engine.AnimationEngine", False, str(e))

try:
    from animation_templates import ANIMATION_TEMPLATES, ARCHETYPE_ANIMATIONS
    test(f"animation_templates ({len(ANIMATION_TEMPLATES)} templates)", True)
except Exception as e:
    test("animation_templates", False, str(e))

try:
    from animation_library import AnimationLibrary
    test("animation_library.AnimationLibrary", True)
except Exception as e:
    test("animation_library.AnimationLibrary", False, str(e))

try:
    from spine_rig_builder import SpineRigBuilder
    test("spine_rig_builder.SpineRigBuilder", True)
except Exception as e:
    test("spine_rig_builder.SpineRigBuilder", False, str(e))

try:
    from precision_segmenter import PrecisionSegmenter
    test("precision_segmenter.PrecisionSegmenter", True)
except Exception as e:
    test("precision_segmenter.PrecisionSegmenter", False, str(e))

try:
    from veilbreakers_rigger import VeilbreakersRigger as CoreRigger
    test("veilbreakers_rigger.VeilbreakersRigger", True)
except Exception as e:
    test("veilbreakers_rigger.VeilbreakersRigger", False, str(e))

try:
    from veilbreakers_cli import VeilbreakersRigger
    test("veilbreakers_cli.VeilbreakersRigger", True)
except Exception as e:
    test("veilbreakers_cli.VeilbreakersRigger", False, str(e))

# 3. Functionality Tests
print("\n[3] FUNCTIONALITY TESTS")
print("-" * 40)

try:
    engine = AnimationEngine()
    test("AnimationEngine instantiation", True)
except Exception as e:
    test("AnimationEngine instantiation", False, str(e))

try:
    lib = AnimationLibrary()
    test("AnimationLibrary instantiation", True)
except Exception as e:
    test("AnimationLibrary instantiation", False, str(e))

try:
    builder = SpineRigBuilder()
    test("SpineRigBuilder instantiation", True)
except Exception as e:
    test("SpineRigBuilder instantiation", False, str(e))

try:
    # Test animation generation
    engine = AnimationEngine()
    anim = engine.create_animation("idle_breathe", "humanoid")
    has_keys = "timelines" in anim or "bones" in str(anim)
    test("Animation generation (idle_breathe)", has_keys)
except Exception as e:
    test("Animation generation", False, str(e))

try:
    # Test Spine export structure
    builder = SpineRigBuilder()
    skeleton = builder.build_skeleton("humanoid")
    has_bones = "bones" in skeleton
    test("Spine skeleton generation", has_bones)
except Exception as e:
    test("Spine skeleton generation", False, str(e))

# 4. SAM2 Check (optional)
print("\n[4] SAM2 (OPTIONAL)")
print("-" * 40)

try:
    import sam2
    test("sam2 module", True)
except Exception as e:
    test("sam2 module (optional)", False, str(e))

# Summary
print("\n" + "=" * 60)
print(f"  RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if errors:
    print("\nERROR DETAILS:")
    for name, error in errors:
        print(f"\n  {name}:")
        print(f"    {error[:200]}")

if failed == 0:
    print("\n  ALL TESTS PASSED - RIGGER IS READY!")
    sys.exit(0)
else:
    print(f"\n  {failed} TESTS FAILED - SEE ERRORS ABOVE")
    sys.exit(1)
