# VEILBREAKERS Monster Rigger v3.0
## The Ultimate Spine Animation Pipeline

**One image → Full Spine rig with 20+ animations → Game-ready in seconds**

---

## 🎯 What's New in v3.0

### Full Spine Integration
- **Complete Spine 4.1 JSON export** with skeleton, bones, slots, skins, and animations
- **IK constraints** for limbs (arms, legs, wings)
- **Physics constraints** for soft parts (hair, cape, tentacles)
- Ready for Spine Editor fine-tuning or direct Godot runtime import

### 10 Creature Archetypes
| Archetype | Description | Animations |
|-----------|-------------|------------|
| **Humanoid** | Bipedal (2 arms, 2 legs) | 15+ |
| **Multi-Arm** | 4-10 arms | 11+ |
| **Quadruped** | Four-legged beasts | 12+ |
| **Serpent** | Snake-like | 11+ |
| **Skeleton** | Undead/bone | 9+ |
| **Floating** | Ghosts, wisps | 10+ |
| **Giant** | Titans, colossi | 11+ |
| **Insectoid** | Spiders, bugs | 10+ |
| **Winged** | Dragons, angels | 14+ |
| **Eldritch** | Cosmic horrors | 9+ |

### 100+ Animation Templates
- **Idles**: breathing, combat stance, menacing, floating, twitching
- **Movement**: walk, run, slither, shamble, drift, scuttle
- **Attacks**: slash, thrust, overhead, flurry, bite, pounce, beam
- **Reactions**: light hit, heavy hit, launch
- **Deaths**: fall forward, fall backward, dissolve, collapse, explode
- **Specials**: charge, release, transform, roar
- **Utility**: spawn, victory, taunt

### Physics-Enabled Soft Parts
- **Hair** (short, long, wild presets)
- **Capes** (light, heavy presets)
- **Tentacles** (slow, fast presets)
- **Tails** (thick, whip presets)
- **Chains, flames, ethereal effects**

### Natural Language Understanding
```
"Rig this demon lord with 6 arms and a flowing cape"
"Create a skeleton warrior with wild hair"
"Animate this dragon with wings and fire"
```

---

## 🚀 Quick Start

### Installation
```bash
pip install numpy pillow opencv-python gradio
```

### One-Click Rig (CLI)
```bash
# Basic humanoid
python veilbreakers_cli.py monster.png

# Dragon with wings and tail
python veilbreakers_cli.py dragon.png --type winged --tail --wings

# Multi-armed demon with cape
python veilbreakers_cli.py demon.png --arms 6 --cape --hair

# Natural language
python veilbreakers_cli.py monster.png --describe "skeleton with flowing cape"
```

### In Python
```python
from veilbreakers_cli import VeilbreakersRigger

rigger = VeilbreakersRigger()

# Natural language mode
outputs = rigger.rig("demon.png", 
    description="demon lord with 6 arms and a flowing cape")

# Explicit mode  
outputs = rigger.rig("dragon.png",
    archetype="winged",
    has_wings=True,
    has_tail=True)
```

### In Godot (with spine-godot)
```gdscript
var skeleton = $SpineSprite
skeleton.animation_state.set_animation("idle_breathe", true)
skeleton.animation_state.add_animation("attack_slash", false, 0)
```

---

## 📁 Package Contents

```
veilbreakers_rigger/
├── veilbreakers_cli.py        # Master CLI & natural language
├── spine_rig_builder.py       # Spine export engine
├── animation_engine.py        # Core animation system
├── animation_templates.py     # 100+ animations
├── veilbreakers_rigger.py     # AI segmentation engine
├── veilbreakers_rigger_ui.py  # Gradio web UI
├── test_animation_system.py   # Test suite (136 tests)
└── docs/QUICKREF.md           # Quick reference
```

---

## 🎮 Events (Gameplay Hooks)
- `hit_frame` - Deal damage
- `screen_shake` - Camera shake
- `death_complete` - Safe to remove
- `charge_ready` - Special attack ready

---

## 📊 Test Results
```
✅ 136/136 tests passing
✅ All 10 archetypes validated  
✅ All animation templates working
✅ Spine JSON export valid
```

---

## 🆓 License
**100% Free for all commercial use**

All dependencies: BSD-3, Apache 2.0, MIT

---

*Built for VEILBREAKERS by Claude* 🐺
