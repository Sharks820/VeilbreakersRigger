# VEILBREAKERS Monster Rigger - Claude Memory

## Project Overview
AI-powered monster rigging tool for game development. Automatically detects body parts in monster images, segments them, and exports rigged sprites for game engines (Godot, Unity).

## Tech Stack
- **Python**: 3.12 (via venv312/)
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4)
- **GPU**: NVIDIA GeForce RTX 4060 Ti (8GB VRAM)
- **AI Models**:
  - Florence-2 PRO (microsoft/Florence-2-large-ft) - Vision-language detection
  - SAM 2.1 - Segmentation
  - GroundingDINO - Object detection
  - LaMa - Inpainting
- **UI**: Gradio 6.x
- **Fine-tuning**: PEFT with LoRA/rsLoRA

## Key Files
| File | Purpose |
|------|---------|
| `run.py` | Main launcher |
| `veilbreakers_rigger.py` | Core rigger class (124KB) |
| `veilbreakers_rigger_ui.py` | Gradio UI with Animation tab |
| `active_learning.py` | AI training UI with corrections |
| `train_florence2_pro.py` | Florence-2 fine-tuning with LoRA |
| `training_metrics.py` | Learning curve visualization |
| `spine_rig_builder.py` | Spine rig generation with bones, IK, physics |
| `animation_engine.py` | Core animation classes and generators |
| `animation_templates.py` | Pre-built animations per archetype |
| `animation_library.py` | Animation library management |
| `LAUNCH_RIGGER.bat` | Windows launcher (uses venv312) |

## Animation System
The Animation tab in the UI generates complete Spine rigs with:
- **Bone Hierarchy**: Proper parent-child relationships
- **IK Constraints**: For arms, legs, tentacles
- **Physics Constraints**: Hair, cape, tentacles with physics sim
- **15-25 Animations**: Per creature archetype (idle, walk, attack, etc.)

### Creature Archetypes
- Humanoid, Multi Arm, Quadruped, Serpent, Skeleton
- Floating, Giant, Insectoid, Winged, Aquatic, Eldritch, Custom

### Key Integration
The animation system uses parts detected by Florence-2 (or user-added) instead of re-detecting. User corrections are preserved:
```python
custom_parts = {part_name: {..., "confidence": 1.0} for part in detected_parts}
builder.build(..., custom_parts=custom_parts)
```

## Critical Fixes Applied

### Body Part Detection (v5.0)
**Problem**: Florence-2 returned full descriptions like "purple cat head" instead of just "head"
**Solution**: Added `extract_body_part_from_label()` in `active_learning.py:95`
```python
BODY_PARTS = ['head', 'body', 'torso', 'arm', 'leg', 'tail', 'wing', 'eye', 'mouth', ...]
def extract_body_part_from_label(label: str) -> str:
    # Extracts body part from full description
    # "purple cat head" -> "head"
```

### Windows DataLoader Crash
**Problem**: NUM_WORKERS > 0 crashes on Windows
**Solution**: `train_florence2_pro.py` - `NUM_WORKERS = 0 if os.name == 'nt' else 4`

### Python 3.14 No CUDA
**Problem**: Python 3.14 too new, PyTorch only had CPU wheels
**Solution**: Created venv312 with Python 3.12, installed PyTorch+cu124

### VRAM Optimization (8GB)
- Gradient checkpointing enabled
- rsLoRA for better scaling
- float16 on CUDA
- Batch size 1-2 for training

## Training Pipeline
1. User uploads monster image
2. Florence-2 detects parts with bounding boxes
3. User corrects any wrong labels
4. Corrections saved to `training_data/`
5. Fine-tune Florence-2 with LoRA (few epochs)
6. A/B comparison: base vs finetuned model

## Launch Instructions
```batch
:: Double-click LAUNCH_RIGGER.bat
:: OR
venv312\Scripts\python.exe run.py
```

## Model Checkpoints
Download required checkpoints to `checkpoints/`:
- SAM 2.1: `sam2.1_hiera_large.pt`
- GroundingDINO: from HuggingFace
- Florence-2: auto-downloads from HuggingFace

## Archive
Old/test files moved to `archive/` instead of deleted.

## User Preferences
- Prefers straightforward execution over excessive questions
- Values working code over lengthy explanations
- Has 8GB VRAM budget - optimize accordingly
- Uses Windows 11

## Version History
- v5.2: Animation system integrated into main UI
- v5.1: Python 3.12 + CUDA setup, cleanup
- v5.0: Body Part Detection Fix + Learning Enhancements
- v4.0: Florence-2 PRO + Unified UI
- v3.4: Critical bug fixes - Florence Pro training + dtype fix
- v3.3: Critical bug fixes - image disappearing, dtype, duplicate methods
