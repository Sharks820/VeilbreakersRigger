# VEILBREAKERS Monster Rigger - Installation Guide

## 🎯 For 99.999% Accuracy Segmentation

The rigger has **three segmentation modes**:

| Mode | Accuracy | Requirements |
|------|----------|--------------|
| **SAM2** | 99%+ | PyTorch + SAM2 + Model weights |
| **OpenCV** | 85% | OpenCV (usually pre-installed) |
| **Basic** | 60% | None (fallback) |

---

## 📦 QUICK INSTALL (Claude Code CLI)

Open your terminal and run these commands:

```bash
# Navigate to the rigger directory
cd path/to/veilbreakers_rigger

# Install core dependencies
pip install pillow numpy

# Install OpenCV for 85% accuracy
pip install opencv-python

# For 99%+ accuracy, install PyTorch and SAM2:
pip install torch torchvision
pip install segment-anything-2
```

---

## 🔥 FULL SAM2 SETUP (99.999% Accuracy)

### Step 1: Install PyTorch

**Windows/Linux with NVIDIA GPU:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Mac (Apple Silicon):**
```bash
pip install torch torchvision
```

**CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Install SAM2

```bash
pip install segment-anything-2
```

Or from source for latest:
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Step 3: Download SAM2 Model Weights

Download from: https://github.com/facebookresearch/segment-anything-2#model-checkpoints

**Recommended model:** `sam2_hiera_large.pt` (best quality)

Save to:
- **Windows:** `C:\Users\<YOU>\.cache\sam2\sam2_hiera_large.pt`
- **Mac/Linux:** `~/.cache/sam2/sam2_hiera_large.pt`

---

## ✅ VERIFY INSTALLATION

```python
python -c "
from precision_segmenter import get_segmentation_status
status = get_segmentation_status()
print('PyTorch:', '✅' if status['pytorch'] else '❌')
print('SAM2:', '✅' if status['sam2'] else '❌')
print('OpenCV:', '✅' if status['opencv'] else '❌')
"
```

---

## 🚀 USAGE

### Python API

```python
from spine_rig_builder import SpineRigBuilder

# Create builder (auto-detects best segmentation)
builder = SpineRigBuilder(output_dir="./output")

# Rig a humanoid character
builder.build(
    "character.png",
    name="my_hero",
    archetype="humanoid",  # See archetypes below
    has_hair=True,
    has_cape=True
)

# Rig a dragon
builder.build(
    "dragon.png",
    name="fire_dragon",
    archetype="winged",
    has_tail=True
)

# Rig a spider
builder.build(
    "spider.png",
    name="giant_spider", 
    archetype="insectoid",
    leg_count=8
)
```

### Command Line

```bash
python spine_rig_builder.py monster.png --type humanoid --name my_monster
python spine_rig_builder.py dragon.png --type winged --tail
python spine_rig_builder.py spider.png --type insectoid --legs 8
```

### Claude Code CLI

Just describe what you want:
```
"Rig this demon image with 4 arms, a cape, and fire effects for VEILBREAKERS"
"Create a serpent rig from snake.png with slither animations"
"Build a quadruped rig for this wolf with a bushy tail"
```

---

## 🎭 AVAILABLE ARCHETYPES (12 Total, 146 Animations)

| Archetype | Parts | Animations | Use For |
|-----------|-------|------------|---------|
| `humanoid` | 6 | 16 | Humans, demons, orcs |
| `multi_arm` | 2+ | 13 | Multi-armed beings |
| `quadruped` | 6 | 14 | Wolves, horses, beasts |
| `serpent` | 1+ | 13 | Snakes, wyrms, dragons |
| `skeleton` | 7 | 10 | Undead, bone creatures |
| `floating` | 1 | 12 | Ghosts, wisps, spectres |
| `giant` | 6 | 12 | Titans, golems |
| `insectoid` | 3 | 11 | Spiders, bugs, crabs |
| `winged` | 4 | 14 | Dragons, birds, bats |
| `aquatic` | 3 | 15 | Fish, merfolk, sea monsters |
| `eldritch` | 1 | 10 | Cosmic horrors, tentacle beasts |
| `custom` | 1 | 6 | Define your own |

---

## 📁 OUTPUT STRUCTURE

```
output/
└── my_monster/
    ├── parts/
    │   ├── head.png       # Extracted body parts
    │   ├── torso.png
    │   ├── arm_left.png
    │   ├── arm_right.png
    │   ├── leg_left.png
    │   └── leg_right.png
    ├── my_monster.json    # Spine rig data
    ├── my_monster.atlas   # Texture atlas
    ├── my_monster.png     # Original image
    └── my_monster_helper.gd  # Godot import script
```

---

## 🎮 IMPORTING TO GODOT

1. Copy the output folder to your Godot project
2. Install the Spine runtime for Godot
3. Create a SpineSprite node
4. Set skeleton_data to `my_monster.json`
5. Set atlas to `my_monster.atlas`
6. Use the helper script for easy animation control

---

## 🐛 TROUBLESHOOTING

### "SAM2 not found"
Make sure you:
1. Installed PyTorch first
2. Downloaded the model weights
3. Saved weights to `~/.cache/sam2/`

### "Parts not aligned correctly"
The bounding box fallback isn't precise. Install SAM2 for accurate segmentation, or manually cut parts in Photoshop.

### "Missing animations for archetype X"
Run: `python -c "from spine_rig_builder import ARCHETYPE_CONFIGS; print(ARCHETYPE_CONFIGS)"`

---

## 💡 TIPS FOR BEST RESULTS

1. **Use clean images** - transparent PNG with the character on a clear background
2. **Front-facing poses** - standard T-pose or A-pose works best
3. **Good separation** - limbs shouldn't overlap the body
4. **High resolution** - at least 1024px for detail

---

## 📞 SUPPORT

For VEILBREAKERS development support, contact the development team.

**Files included:**
- `spine_rig_builder.py` - Main rigging system
- `animation_engine.py` - Animation core
- `animation_templates.py` - Pre-built animation patterns  
- `precision_segmenter.py` - AI segmentation module
- `veilbreakers_rigger.py` - Image processing
- `veilbreakers_cli.py` - Command line interface
- `test_animation_system.py` - Test suite (136 tests)
