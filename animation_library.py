#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                      VEILBREAKERS ANIMATION LIBRARY                          ║
║                                                                              ║
║   Easy-to-use animation library with:                                        ║
║   • Pre-built animations you can pull directly                               ║
║   • Simple JSON format for custom animations                                 ║
║   • One-line add to any rig                                                  ║
║                                                                              ║
║   Usage:                                                                     ║
║   >>> from animation_library import AnimationLibrary                         ║
║   >>> lib = AnimationLibrary()                                               ║
║   >>> lib.list_all()  # See all available                                    ║
║   >>> lib.add_to_rig(rig, "jump")  # Add jump to your rig                   ║
║   >>> lib.add_custom("my_anim.json")  # Add your own                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum, auto

# =============================================================================
# ANIMATION CATEGORIES
# =============================================================================

class AnimCategory(Enum):
    IDLE = auto()
    MOVEMENT = auto()
    COMBAT = auto()
    DAMAGE = auto()
    DEATH = auto()
    SPECIAL = auto()
    EMOTE = auto()
    PHYSICS = auto()


# =============================================================================
# PRE-BUILT ANIMATION LIBRARY
# =============================================================================

ANIMATION_LIBRARY = {
    # =========================================================================
    # IDLE ANIMATIONS
    # =========================================================================
    "idle_breathe": {
        "category": "IDLE",
        "duration": 2.0,
        "loop": True,
        "description": "Subtle breathing motion",
        "bones": {
            "torso": {"rotate": [(0, 0), (1.0, 2), (2.0, 0)], "translate": [(0, 0, 0), (1.0, 0, -3), (2.0, 0, 0)]},
            "head": {"rotate": [(0, 0), (1.0, 1), (2.0, 0)]},
            "arm_left": {"rotate": [(0, 0), (1.0, 2), (2.0, 0)]},
            "arm_right": {"rotate": [(0, 0), (1.0, -2), (2.0, 0)]},
        }
    },
    
    "idle_combat": {
        "category": "IDLE",
        "duration": 1.5,
        "loop": True,
        "description": "Ready combat stance with slight bounce",
        "bones": {
            "torso": {"rotate": [(0, -5), (0.75, -3), (1.5, -5)], "translate": [(0, 0, 0), (0.75, 0, -5), (1.5, 0, 0)]},
            "head": {"rotate": [(0, 5), (0.75, 8), (1.5, 5)]},
            "arm_left": {"rotate": [(0, -30), (0.75, -25), (1.5, -30)], "translate": [(0, 0, -20), (0.75, 0, -25), (1.5, 0, -20)]},
            "arm_right": {"rotate": [(0, 45), (0.75, 50), (1.5, 45)], "translate": [(0, 0, -30), (0.75, 0, -35), (1.5, 0, -30)]},
        }
    },
    
    "idle_tired": {
        "category": "IDLE",
        "duration": 3.0,
        "loop": True,
        "description": "Exhausted, heavy breathing",
        "bones": {
            "torso": {"rotate": [(0, 10), (1.5, 15), (3.0, 10)], "translate": [(0, 0, 10), (1.5, 0, 20), (3.0, 0, 10)]},
            "head": {"rotate": [(0, 15), (1.5, 20), (3.0, 15)]},
            "arm_left": {"rotate": [(0, 10), (1.5, 5), (3.0, 10)]},
            "arm_right": {"rotate": [(0, -10), (1.5, -5), (3.0, -10)]},
        }
    },
    
    "idle_look_around": {
        "category": "IDLE",
        "duration": 4.0,
        "loop": True,
        "description": "Looking around alertly",
        "bones": {
            "head": {"rotate": [(0, 0), (1.0, -30), (2.0, 0), (3.0, 30), (4.0, 0)]},
            "torso": {"rotate": [(0, 0), (1.0, -5), (2.0, 0), (3.0, 5), (4.0, 0)]},
        }
    },

    # =========================================================================
    # MOVEMENT ANIMATIONS
    # =========================================================================
    "walk": {
        "category": "MOVEMENT",
        "duration": 0.8,
        "loop": True,
        "description": "Standard walking cycle",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.2, 0, -8), (0.4, 0, 0), (0.6, 0, -8), (0.8, 0, 0)], "rotate": [(0, -2), (0.2, 2), (0.4, -2), (0.6, 2), (0.8, -2)]},
            "head": {"translate": [(0, 0, 0), (0.2, 0, -5), (0.4, 0, 0), (0.6, 0, -5), (0.8, 0, 0)]},
            "arm_left": {"rotate": [(0, 20), (0.4, -20), (0.8, 20)]},
            "arm_right": {"rotate": [(0, -20), (0.4, 20), (0.8, -20)]},
            "leg_left": {"rotate": [(0, -15), (0.4, 15), (0.8, -15)], "translate": [(0, 0, 0), (0.4, 0, -10), (0.8, 0, 0)]},
            "leg_right": {"rotate": [(0, 15), (0.4, -15), (0.8, 15)], "translate": [(0, 0, -10), (0.4, 0, 0), (0.8, 0, -10)]},
        }
    },
    
    "run": {
        "category": "MOVEMENT",
        "duration": 0.5,
        "loop": True,
        "description": "Fast running cycle",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.125, 0, -15), (0.25, 0, 0), (0.375, 0, -15), (0.5, 0, 0)], "rotate": [(0, -8), (0.125, -5), (0.25, -8), (0.375, -5), (0.5, -8)]},
            "head": {"rotate": [(0, 8), (0.25, 5), (0.5, 8)]},
            "arm_left": {"rotate": [(0, 45), (0.25, -45), (0.5, 45)]},
            "arm_right": {"rotate": [(0, -45), (0.25, 45), (0.5, -45)]},
            "leg_left": {"rotate": [(0, -30), (0.25, 30), (0.5, -30)], "translate": [(0, 0, 0), (0.25, 0, -20), (0.5, 0, 0)]},
            "leg_right": {"rotate": [(0, 30), (0.25, -30), (0.5, 30)], "translate": [(0, 0, -20), (0.25, 0, 0), (0.5, 0, -20)]},
        }
    },
    
    "sprint": {
        "category": "MOVEMENT",
        "duration": 0.35,
        "loop": True,
        "description": "Maximum speed sprint",
        "bones": {
            "torso": {"rotate": [(0, -15), (0.175, -12), (0.35, -15)], "translate": [(0, 0, 0), (0.175, 0, -20), (0.35, 0, 0)]},
            "head": {"rotate": [(0, 15), (0.35, 15)]},
            "arm_left": {"rotate": [(0, 60), (0.175, -60), (0.35, 60)]},
            "arm_right": {"rotate": [(0, -60), (0.175, 60), (0.35, -60)]},
            "leg_left": {"rotate": [(0, -45), (0.175, 45), (0.35, -45)]},
            "leg_right": {"rotate": [(0, 45), (0.175, -45), (0.35, 45)]},
        }
    },
    
    "jump": {
        "category": "MOVEMENT",
        "duration": 0.8,
        "loop": False,
        "description": "Jump with anticipation and landing",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.15, 0, 20), (0.3, 0, -80), (0.5, 0, -100), (0.7, 0, -20), (0.8, 0, 0)], "scale": [(0, 1, 1), (0.15, 1, 0.9), (0.3, 1, 1.1), (0.8, 1, 1)]},
            "arm_left": {"rotate": [(0, 0), (0.15, 20), (0.3, -120), (0.5, -150), (0.7, -30), (0.8, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.15, -20), (0.3, 120), (0.5, 150), (0.7, 30), (0.8, 0)]},
            "leg_left": {"rotate": [(0, 0), (0.15, 30), (0.3, -30), (0.5, -20), (0.7, 20), (0.8, 0)]},
            "leg_right": {"rotate": [(0, 0), (0.15, 30), (0.3, -30), (0.5, -20), (0.7, 20), (0.8, 0)]},
        }
    },
    
    "dodge_left": {
        "category": "MOVEMENT",
        "duration": 0.4,
        "loop": False,
        "description": "Quick dodge to the left",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.1, -50, -10), (0.3, -80, 0), (0.4, 0, 0)], "rotate": [(0, 0), (0.2, 20), (0.4, 0)]},
            "head": {"rotate": [(0, 0), (0.2, -15), (0.4, 0)]},
        }
    },
    
    "dodge_right": {
        "category": "MOVEMENT",
        "duration": 0.4,
        "loop": False,
        "description": "Quick dodge to the right",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.1, 50, -10), (0.3, 80, 0), (0.4, 0, 0)], "rotate": [(0, 0), (0.2, -20), (0.4, 0)]},
            "head": {"rotate": [(0, 0), (0.2, 15), (0.4, 0)]},
        }
    },
    
    "dodge_back": {
        "category": "MOVEMENT",
        "duration": 0.5,
        "loop": False,
        "description": "Backstep dodge",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.15, 0, -20), (0.35, 0, 60), (0.5, 0, 0)], "rotate": [(0, 0), (0.2, -15), (0.5, 0)]},
            "leg_left": {"rotate": [(0, 0), (0.25, -30), (0.5, 0)]},
            "leg_right": {"rotate": [(0, 0), (0.25, -30), (0.5, 0)]},
        }
    },
    
    "crouch": {
        "category": "MOVEMENT",
        "duration": 0.3,
        "loop": False,
        "description": "Crouch down",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.3, 0, 60)], "scale": [(0, 1, 1), (0.3, 1, 0.8)]},
            "head": {"rotate": [(0, 0), (0.3, 10)]},
            "leg_left": {"rotate": [(0, 0), (0.3, 45)]},
            "leg_right": {"rotate": [(0, 0), (0.3, 45)]},
        }
    },
    
    "crouch_walk": {
        "category": "MOVEMENT",
        "duration": 1.2,
        "loop": True,
        "description": "Sneaky crouch walking",
        "bones": {
            "torso": {"translate": [(0, 0, 50), (0.6, 0, 55), (1.2, 0, 50)], "rotate": [(0, 10), (0.6, 12), (1.2, 10)]},
            "leg_left": {"rotate": [(0, 30), (0.6, 50), (1.2, 30)]},
            "leg_right": {"rotate": [(0, 50), (0.6, 30), (1.2, 50)]},
        }
    },

    # =========================================================================
    # COMBAT ANIMATIONS
    # =========================================================================
    "attack_slash": {
        "category": "COMBAT",
        "duration": 0.6,
        "loop": False,
        "description": "Horizontal sword slash",
        "bones": {
            "torso": {"rotate": [(0, 0), (0.15, -20), (0.3, 30), (0.6, 0)]},
            "head": {"rotate": [(0, 0), (0.15, -10), (0.3, 15), (0.6, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.15, -120), (0.3, 60), (0.6, 0)], "translate": [(0, 0, 0), (0.15, 0, -50), (0.3, 0, 20), (0.6, 0, 0)]},
            "arm_left": {"rotate": [(0, 0), (0.15, -20), (0.3, 10), (0.6, 0)]},
        }
    },
    
    "attack_thrust": {
        "category": "COMBAT",
        "duration": 0.5,
        "loop": False,
        "description": "Forward thrust/stab",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.2, -20, 0), (0.35, 40, 0), (0.5, 0, 0)], "rotate": [(0, 0), (0.2, -10), (0.35, 5), (0.5, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.2, -60), (0.35, 10), (0.5, 0)], "translate": [(0, 0, 0), (0.2, -30, 0), (0.35, 80, 0), (0.5, 0, 0)]},
        }
    },
    
    "attack_overhead": {
        "category": "COMBAT",
        "duration": 0.7,
        "loop": False,
        "description": "Powerful overhead strike",
        "bones": {
            "torso": {"scale": [(0, 1, 1), (0.3, 1, 0.95), (0.5, 1, 1.05), (0.7, 1, 1)], "translate": [(0, 0, 0), (0.3, 0, 10), (0.5, 0, -20), (0.7, 0, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.3, -180), (0.5, 90), (0.7, 0)]},
            "arm_left": {"rotate": [(0, 0), (0.3, -160), (0.5, 70), (0.7, 0)]},
            "head": {"rotate": [(0, 0), (0.3, -20), (0.5, 30), (0.7, 0)]},
        }
    },
    
    "attack_uppercut": {
        "category": "COMBAT",
        "duration": 0.5,
        "loop": False,
        "description": "Rising uppercut punch",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.2, 0, 15), (0.35, 0, -30), (0.5, 0, 0)], "rotate": [(0, 0), (0.2, 10), (0.35, -15), (0.5, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.2, 60), (0.35, -120), (0.5, 0)], "translate": [(0, 0, 0), (0.2, 0, 20), (0.35, 0, -60), (0.5, 0, 0)]},
        }
    },
    
    "attack_kick": {
        "category": "COMBAT",
        "duration": 0.5,
        "loop": False,
        "description": "Front kick",
        "bones": {
            "torso": {"rotate": [(0, 0), (0.2, -10), (0.35, 5), (0.5, 0)]},
            "leg_right": {"rotate": [(0, 0), (0.15, 30), (0.3, -90), (0.5, 0)], "translate": [(0, 0, 0), (0.3, 50, -30), (0.5, 0, 0)]},
        }
    },
    
    "attack_spin": {
        "category": "COMBAT",
        "duration": 0.8,
        "loop": False,
        "description": "360 spin attack",
        "bones": {
            "torso": {"rotate": [(0, 0), (0.4, 180), (0.8, 360)]},
            "arm_right": {"rotate": [(0, 0), (0.2, -90), (0.6, 90), (0.8, 0)]},
            "arm_left": {"rotate": [(0, 0), (0.2, 90), (0.6, -90), (0.8, 0)]},
        }
    },
    
    "attack_combo_1": {
        "category": "COMBAT",
        "duration": 1.2,
        "loop": False,
        "description": "3-hit combo: slash, slash, thrust",
        "bones": {
            "torso": {"rotate": [(0, 0), (0.15, -15), (0.3, 15), (0.5, -15), (0.7, 15), (0.9, -10), (1.0, 5), (1.2, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.15, -90), (0.3, 45), (0.5, -90), (0.7, 45), (0.9, -60), (1.0, 10), (1.2, 0)], "translate": [(0, 0, 0), (0.9, -20, 0), (1.0, 80, 0), (1.2, 0, 0)]},
        }
    },
    
    "block": {
        "category": "COMBAT",
        "duration": 0.2,
        "loop": False,
        "description": "Raise shield/weapon to block",
        "bones": {
            "torso": {"rotate": [(0, 0), (0.2, -5)]},
            "arm_left": {"rotate": [(0, 0), (0.2, -90)], "translate": [(0, 0, 0), (0.2, 30, -20)]},
        }
    },
    
    "block_hit": {
        "category": "COMBAT",
        "duration": 0.3,
        "loop": False,
        "description": "Recoil from blocked hit",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.1, 20, 0), (0.3, 0, 0)], "rotate": [(0, -5), (0.1, 5), (0.3, -5)]},
            "arm_left": {"rotate": [(0, -90), (0.1, -70), (0.3, -90)]},
        }
    },
    
    "parry": {
        "category": "COMBAT",
        "duration": 0.4,
        "loop": False,
        "description": "Deflect attack with weapon",
        "bones": {
            "torso": {"rotate": [(0, 0), (0.1, 10), (0.25, -20), (0.4, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.1, -30), (0.25, 60), (0.4, 0)]},
        }
    },

    # =========================================================================
    # DAMAGE ANIMATIONS
    # =========================================================================
    "hit_light": {
        "category": "DAMAGE",
        "duration": 0.4,
        "loop": False,
        "description": "Light hit reaction",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.1, 30, 0), (0.4, 0, 0)], "rotate": [(0, 0), (0.1, 10), (0.4, 0)]},
            "head": {"rotate": [(0, 0), (0.1, 20), (0.4, 0)]},
        }
    },
    
    "hit_heavy": {
        "category": "DAMAGE",
        "duration": 0.8,
        "loop": False,
        "description": "Heavy hit with stumble",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.15, 50, 10), (0.4, 30, 5), (0.8, 0, 0)], "rotate": [(0, 0), (0.15, 20), (0.4, 15), (0.8, 0)]},
            "head": {"rotate": [(0, 0), (0.15, 35), (0.8, 0)]},
            "arm_left": {"rotate": [(0, 0), (0.15, 30), (0.8, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.15, -20), (0.8, 0)]},
        }
    },
    
    "hit_back": {
        "category": "DAMAGE",
        "duration": 0.5,
        "loop": False,
        "description": "Hit from behind",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.15, -30, 0), (0.5, 0, 0)], "rotate": [(0, 0), (0.15, -15), (0.5, 0)]},
            "head": {"rotate": [(0, 0), (0.15, -25), (0.5, 0)]},
        }
    },
    
    "knockdown": {
        "category": "DAMAGE",
        "duration": 1.0,
        "loop": False,
        "description": "Knocked to the ground",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.3, 60, 50), (0.6, 80, 150), (1.0, 80, 180)], "rotate": [(0, 0), (0.3, 30), (0.6, 60), (1.0, 80)]},
            "head": {"rotate": [(0, 0), (0.3, 40), (1.0, 70)]},
            "arm_left": {"rotate": [(0, 0), (0.6, 90), (1.0, 120)]},
            "arm_right": {"rotate": [(0, 0), (0.6, -60), (1.0, -90)]},
            "leg_left": {"rotate": [(0, 0), (1.0, -20)]},
            "leg_right": {"rotate": [(0, 0), (1.0, 30)]},
        }
    },
    
    "get_up": {
        "category": "DAMAGE",
        "duration": 1.2,
        "loop": False,
        "description": "Get up from knockdown",
        "bones": {
            "torso": {"translate": [(0, 80, 180), (0.4, 60, 100), (0.8, 20, 30), (1.2, 0, 0)], "rotate": [(0, 80), (0.4, 40), (0.8, 10), (1.2, 0)]},
            "head": {"rotate": [(0, 70), (0.8, 10), (1.2, 0)]},
            "arm_left": {"rotate": [(0, 120), (0.4, 60), (1.2, 0)]},
            "arm_right": {"rotate": [(0, -90), (0.4, -40), (1.2, 0)]},
        }
    },
    
    "stagger": {
        "category": "DAMAGE",
        "duration": 0.6,
        "loop": False,
        "description": "Stagger backwards",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.2, 40, 20), (0.4, 30, 10), (0.6, 0, 0)], "rotate": [(0, 0), (0.2, 15), (0.6, 0)]},
            "leg_left": {"rotate": [(0, 0), (0.3, -20), (0.6, 0)]},
            "leg_right": {"rotate": [(0, 0), (0.15, 20), (0.45, -10), (0.6, 0)]},
        }
    },

    # =========================================================================
    # DEATH ANIMATIONS
    # =========================================================================
    "death_fall_forward": {
        "category": "DEATH",
        "duration": 1.2,
        "loop": False,
        "description": "Fall forward death",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.4, 0, 20), (0.8, 0, 100), (1.2, 0, 200)], "rotate": [(0, 0), (0.4, 30), (0.8, 70), (1.2, 90)]},
            "head": {"rotate": [(0, 0), (0.4, 40), (1.2, 90)]},
            "arm_left": {"rotate": [(0, 0), (0.6, 120), (1.2, 180)]},
            "arm_right": {"rotate": [(0, 0), (0.6, -90), (1.2, -120)]},
            "leg_left": {"rotate": [(0, 0), (1.2, -30)]},
            "leg_right": {"rotate": [(0, 0), (1.2, 30)]},
        }
    },
    
    "death_fall_backward": {
        "category": "DEATH",
        "duration": 1.2,
        "loop": False,
        "description": "Fall backward death",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.4, 0, 20), (1.2, 0, 180)], "rotate": [(0, 0), (0.4, -30), (1.2, -90)]},
            "head": {"rotate": [(0, 0), (1.2, -60)]},
            "arm_left": {"rotate": [(0, 0), (1.2, -90)]},
            "arm_right": {"rotate": [(0, 0), (1.2, 90)]},
        }
    },
    
    "death_explode": {
        "category": "DEATH",
        "duration": 0.5,
        "loop": False,
        "description": "Explosive death (for magical/elemental)",
        "bones": {
            "torso": {"scale": [(0, 1, 1), (0.2, 1.3, 1.3), (0.5, 0, 0)]},
            "head": {"scale": [(0, 1, 1), (0.15, 1.5, 1.5), (0.4, 0, 0)], "translate": [(0, 0, 0), (0.4, 0, -100)]},
            "arm_left": {"rotate": [(0, 0), (0.3, -120)], "scale": [(0, 1, 1), (0.5, 0, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.3, 120)], "scale": [(0, 1, 1), (0.5, 0, 0)]},
        }
    },
    
    "death_dissolve": {
        "category": "DEATH",
        "duration": 2.0,
        "loop": False,
        "description": "Fade away death (for ghosts/magic)",
        "bones": {
            "torso": {"scale": [(0, 1, 1), (2.0, 0.8, 1.2)], "translate": [(0, 0, 0), (2.0, 0, -50)]},
            "head": {"translate": [(0, 0, 0), (2.0, 0, -30)]},
        }
    },
    
    "death_crumble": {
        "category": "DEATH",
        "duration": 1.5,
        "loop": False,
        "description": "Crumble death (for skeletons/golems)",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.5, 0, 30), (1.5, 0, 200)], "rotate": [(0, 0), (0.5, 20), (1.0, 45)]},
            "head": {"translate": [(0, 0, 0), (0.3, 20, -20), (1.0, 80, 100)], "rotate": [(0, 0), (1.0, 180)]},
            "arm_left": {"translate": [(0, 0, 0), (0.4, -40, 30), (1.2, -100, 150)]},
            "arm_right": {"translate": [(0, 0, 0), (0.4, 40, 30), (1.2, 100, 150)]},
        }
    },

    # =========================================================================
    # SPECIAL ANIMATIONS
    # =========================================================================
    "special_charge": {
        "category": "SPECIAL",
        "duration": 1.0,
        "loop": True,
        "description": "Charging up power",
        "bones": {
            "torso": {"scale": [(0, 1, 1), (0.5, 1.05, 0.95), (1.0, 1, 1)]},
            "head": {"rotate": [(0, -5), (0.5, 5), (1.0, -5)]},
            "arm_left": {"rotate": [(0, -20), (0.5, -30), (1.0, -20)]},
            "arm_right": {"rotate": [(0, 20), (0.5, 30), (1.0, 20)]},
        }
    },
    
    "special_release": {
        "category": "SPECIAL",
        "duration": 0.6,
        "loop": False,
        "description": "Release charged power",
        "bones": {
            "torso": {"scale": [(0, 1, 1), (0.2, 1.2, 0.8), (0.6, 1, 1)]},
            "head": {"rotate": [(0, 0), (0.2, -30), (0.6, 0)]},
            "arm_left": {"rotate": [(0, 0), (0.2, -90), (0.6, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.2, 90), (0.6, 0)]},
        }
    },
    
    "cast_spell": {
        "category": "SPECIAL",
        "duration": 0.8,
        "loop": False,
        "description": "Cast a spell with hand gesture",
        "bones": {
            "torso": {"rotate": [(0, 0), (0.3, -10), (0.8, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.2, -60), (0.5, -90), (0.8, 0)], "translate": [(0, 0, 0), (0.3, 40, -30), (0.8, 0, 0)]},
            "head": {"rotate": [(0, 0), (0.3, 10), (0.8, 0)]},
        }
    },
    
    "summon": {
        "category": "SPECIAL",
        "duration": 1.5,
        "loop": False,
        "description": "Summoning animation",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.5, 0, -20), (1.5, 0, 0)], "rotate": [(0, 0), (0.75, 5), (1.5, 0)]},
            "arm_left": {"rotate": [(0, 0), (0.5, -120), (1.0, -150), (1.5, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.5, 120), (1.0, 150), (1.5, 0)]},
            "head": {"rotate": [(0, 0), (0.5, -20), (1.0, -30), (1.5, 0)]},
        }
    },
    
    "transform": {
        "category": "SPECIAL",
        "duration": 1.0,
        "loop": False,
        "description": "Transformation/morph animation",
        "bones": {
            "torso": {"scale": [(0, 1, 1), (0.3, 0.8, 1.2), (0.6, 1.2, 0.8), (1.0, 1, 1)]},
            "head": {"scale": [(0, 1, 1), (0.5, 1.3, 1.3), (1.0, 1, 1)]},
            "arm_left": {"scale": [(0, 1, 1), (0.4, 1.5, 1.5), (1.0, 1, 1)]},
            "arm_right": {"scale": [(0, 1, 1), (0.4, 1.5, 1.5), (1.0, 1, 1)]},
        }
    },
    
    "teleport_out": {
        "category": "SPECIAL",
        "duration": 0.4,
        "loop": False,
        "description": "Teleport disappear",
        "bones": {
            "torso": {"scale": [(0, 1, 1), (0.2, 1.2, 0.5), (0.4, 0, 2)], "translate": [(0, 0, 0), (0.4, 0, -100)]},
        }
    },
    
    "teleport_in": {
        "category": "SPECIAL",
        "duration": 0.4,
        "loop": False,
        "description": "Teleport appear",
        "bones": {
            "torso": {"scale": [(0, 0, 2), (0.2, 1.2, 0.5), (0.4, 1, 1)], "translate": [(0, 0, -100), (0.4, 0, 0)]},
        }
    },

    # =========================================================================
    # EMOTE ANIMATIONS
    # =========================================================================
    "spawn": {
        "category": "EMOTE",
        "duration": 1.5,
        "loop": False,
        "description": "Character spawn/appear animation",
        "bones": {
            "torso": {"scale": [(0, 0, 0), (0.75, 1.1, 1.1), (1.5, 1, 1)]},
            "head": {"scale": [(0, 0, 0), (0.9, 1.1, 1.1), (1.5, 1, 1)]},
            "arm_left": {"scale": [(0, 0, 1), (1.0, 1, 1)]},
            "arm_right": {"scale": [(0, 0, 1), (1.0, 1, 1)]},
            "leg_left": {"scale": [(0, 1, 0), (1.2, 1, 1)]},
            "leg_right": {"scale": [(0, 1, 0), (1.2, 1, 1)]},
        }
    },
    
    "victory": {
        "category": "EMOTE",
        "duration": 2.0,
        "loop": False,
        "description": "Victory celebration",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.5, 0, -20), (2.0, 0, 0)]},
            "arm_left": {"rotate": [(0, 0), (0.5, -150), (1.5, -150), (2.0, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.5, 150), (1.5, 150), (2.0, 0)]},
            "head": {"rotate": [(0, 0), (0.5, -10), (2.0, 0)]},
        }
    },
    
    "taunt": {
        "category": "EMOTE",
        "duration": 1.2,
        "loop": False,
        "description": "Taunt/mock enemy",
        "bones": {
            "torso": {"rotate": [(0, 0), (0.3, -10), (0.6, 10), (0.9, -10), (1.2, 0)]},
            "head": {"rotate": [(0, 0), (0.3, 15), (0.6, -15), (1.2, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.2, 30), (0.4, -20), (0.6, 30), (1.2, 0)]},
        }
    },
    
    "wave": {
        "category": "EMOTE",
        "duration": 1.0,
        "loop": False,
        "description": "Friendly wave",
        "bones": {
            "arm_right": {"rotate": [(0, 0), (0.2, -120), (0.4, -100), (0.6, -120), (0.8, -100), (1.0, 0)]},
            "head": {"rotate": [(0, 0), (0.5, 10), (1.0, 0)]},
        }
    },
    
    "bow": {
        "category": "EMOTE",
        "duration": 1.5,
        "loop": False,
        "description": "Respectful bow",
        "bones": {
            "torso": {"rotate": [(0, 0), (0.5, 45), (1.0, 45), (1.5, 0)]},
            "head": {"rotate": [(0, 0), (0.5, 30), (1.0, 30), (1.5, 0)]},
            "arm_left": {"rotate": [(0, 0), (0.5, 20), (1.5, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.5, -20), (1.5, 0)]},
        }
    },
    
    "laugh": {
        "category": "EMOTE",
        "duration": 1.5,
        "loop": False,
        "description": "Laughing animation",
        "bones": {
            "torso": {"translate": [(0, 0, 0), (0.15, 0, -5), (0.3, 0, 0), (0.45, 0, -5), (0.6, 0, 0), (0.75, 0, -5), (0.9, 0, 0)]},
            "head": {"rotate": [(0, 0), (0.15, -10), (0.3, 10), (0.45, -10), (0.6, 10), (0.75, -5), (1.5, 0)]},
        }
    },
    
    "point": {
        "category": "EMOTE",
        "duration": 0.8,
        "loop": False,
        "description": "Point at something",
        "bones": {
            "arm_right": {"rotate": [(0, 0), (0.3, -80), (0.6, -80), (0.8, 0)]},
            "head": {"rotate": [(0, 0), (0.3, 15), (0.6, 15), (0.8, 0)]},
            "torso": {"rotate": [(0, 0), (0.3, -10), (0.6, -10), (0.8, 0)]},
        }
    },
    
    "shrug": {
        "category": "EMOTE",
        "duration": 0.8,
        "loop": False,
        "description": "Shrug shoulders",
        "bones": {
            "arm_left": {"rotate": [(0, 0), (0.3, -30), (0.5, -30), (0.8, 0)], "translate": [(0, 0, 0), (0.3, 0, -20), (0.5, 0, -20), (0.8, 0, 0)]},
            "arm_right": {"rotate": [(0, 0), (0.3, 30), (0.5, 30), (0.8, 0)], "translate": [(0, 0, 0), (0.3, 0, -20), (0.5, 0, -20), (0.8, 0, 0)]},
            "head": {"rotate": [(0, 0), (0.3, 10), (0.5, 10), (0.8, 0)]},
        }
    },

    # =========================================================================
    # PHYSICS ANIMATIONS
    # =========================================================================
    "physics_hair": {
        "category": "PHYSICS",
        "duration": 3.0,
        "loop": True,
        "description": "Hair physics simulation",
        "bones": {
            "hair_1": {"rotate": [(0, 0), (0.75, 10), (1.5, -10), (2.25, 10), (3.0, 0)]},
            "hair_2": {"rotate": [(0, 0), (0.5, -8), (1.0, 8), (1.5, -8), (2.0, 8), (2.5, -8), (3.0, 0)]},
            "hair_3": {"rotate": [(0, 0), (1.0, 12), (2.0, -12), (3.0, 0)]},
        }
    },
    
    "physics_cape": {
        "category": "PHYSICS",
        "duration": 2.0,
        "loop": True,
        "description": "Cape physics simulation",
        "bones": {
            "cape": {"rotate": [(0, 0), (0.5, 8), (1.0, -8), (1.5, 8), (2.0, 0)]},
            "cape_2": {"rotate": [(0, 0), (0.4, 12), (0.8, -12), (1.2, 12), (1.6, -12), (2.0, 0)]},
            "cape_3": {"rotate": [(0, 0), (0.3, 15), (0.7, -15), (1.1, 15), (1.5, -15), (2.0, 0)]},
        }
    },
    
    "physics_tail": {
        "category": "PHYSICS",
        "duration": 2.5,
        "loop": True,
        "description": "Tail wagging/physics",
        "bones": {
            "tail": {"rotate": [(0, 0), (0.625, 20), (1.25, -20), (1.875, 20), (2.5, 0)]},
            "tail_2": {"rotate": [(0, 0), (0.5, 30), (1.0, -30), (1.5, 30), (2.0, -30), (2.5, 0)]},
        }
    },
    
    "physics_tentacle": {
        "category": "PHYSICS",
        "duration": 2.0,
        "loop": True,
        "description": "Tentacle wave physics",
        "bones": {
            "tentacle_1": {"rotate": [(0, 0), (0.5, 15), (1.0, -15), (1.5, 15), (2.0, 0)]},
            "tentacle_2": {"rotate": [(0, 0), (0.4, -20), (0.8, 20), (1.2, -20), (1.6, 20), (2.0, 0)]},
            "tentacle_3": {"rotate": [(0, 0), (0.33, 25), (0.66, -25), (1.0, 25), (1.33, -25), (1.66, 25), (2.0, 0)]},
        }
    },
}


# =============================================================================
# ANIMATION LIBRARY CLASS
# =============================================================================

class AnimationLibrary:
    """
    Easy-to-use animation library for VEILBREAKERS.
    
    Usage:
        lib = AnimationLibrary()
        lib.list_all()                    # See all animations
        lib.list_category("COMBAT")       # See combat animations
        lib.get("attack_slash")           # Get animation data
        lib.add_to_rig(rig, "jump")       # Add to a rig
        lib.add_custom("my_anim.json")    # Add custom animation
    """
    
    def __init__(self, custom_dir: str = "./custom_animations"):
        self.library = ANIMATION_LIBRARY.copy()
        self.custom_dir = Path(custom_dir)
        
        # Load any custom animations
        if self.custom_dir.exists():
            self._load_custom_animations()
    
    def _load_custom_animations(self):
        """Load custom animations from JSON files"""
        for json_file in self.custom_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    anim_data = json.load(f)
                
                name = json_file.stem
                self.library[name] = anim_data
                print(f"  Loaded custom animation: {name}")
            except Exception as e:
                print(f"  Failed to load {json_file}: {e}")
    
    def list_all(self) -> List[str]:
        """List all available animations"""
        print("\n" + "=" * 60)
        print("  ANIMATION LIBRARY")
        print("=" * 60)
        
        by_category = {}
        for name, data in self.library.items():
            cat = data.get("category", "MISC")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((name, data.get("description", "")))
        
        for cat in sorted(by_category.keys()):
            print(f"\n  {cat} ({len(by_category[cat])} animations):")
            for name, desc in sorted(by_category[cat]):
                print(f"    • {name}: {desc}")
        
        print(f"\n  TOTAL: {len(self.library)} animations available")
        print("=" * 60)
        
        return list(self.library.keys())
    
    def list_category(self, category: str) -> List[str]:
        """List animations in a specific category"""
        results = []
        for name, data in self.library.items():
            if data.get("category", "").upper() == category.upper():
                results.append(name)
                print(f"  • {name}: {data.get('description', '')}")
        return results
    
    def get(self, name: str) -> Optional[Dict]:
        """Get animation data by name"""
        return self.library.get(name)
    
    def search(self, keyword: str) -> List[str]:
        """Search animations by keyword"""
        results = []
        keyword = keyword.lower()
        
        for name, data in self.library.items():
            if keyword in name.lower() or keyword in data.get("description", "").lower():
                results.append(name)
        
        return results
    
    def add_custom(self, json_path: str) -> bool:
        """
        Add a custom animation from a JSON file.
        
        JSON format:
        {
            "category": "COMBAT",
            "duration": 0.6,
            "loop": false,
            "description": "My custom attack",
            "bones": {
                "torso": {"rotate": [[0, 0], [0.3, 20], [0.6, 0]]},
                "arm_right": {"rotate": [[0, 0], [0.3, -90], [0.6, 0]]}
            }
        }
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            name = Path(json_path).stem
            self.library[name] = data
            
            # Save to custom directory
            self.custom_dir.mkdir(exist_ok=True)
            save_path = self.custom_dir / f"{name}.json"
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Added custom animation: {name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add animation: {e}")
            return False
    
    def add_to_rig(self, rig, animation_name: str) -> bool:
        """Add an animation from the library to a rig"""
        if animation_name not in self.library:
            print(f"❌ Animation '{animation_name}' not found in library")
            return False
        
        anim_data = self.library[animation_name]
        
        # Convert library format to Spine format
        try:
            from animation_engine import Animation
            
            anim = Animation(
                name=animation_name,
                duration=anim_data.get("duration", 1.0),
                loop=anim_data.get("loop", False)
            )
            
            # Add bone timelines
            for bone_name, timelines in anim_data.get("bones", {}).items():
                for timeline_type, keyframes in timelines.items():
                    if timeline_type == "rotate":
                        for kf in keyframes:
                            time, value = kf[0], kf[1]
                            anim.add_bone_keyframe(bone_name, "rotate", time, value)
                    elif timeline_type == "translate":
                        for kf in keyframes:
                            time = kf[0]
                            x = kf[1] if len(kf) > 1 else 0
                            y = kf[2] if len(kf) > 2 else 0
                            anim.add_bone_keyframe(bone_name, "translate", time, {"x": x, "y": y})
                    elif timeline_type == "scale":
                        for kf in keyframes:
                            time = kf[0]
                            sx = kf[1] if len(kf) > 1 else 1
                            sy = kf[2] if len(kf) > 2 else 1
                            anim.add_bone_keyframe(bone_name, "scale", time, {"x": sx, "y": sy})
            
            rig.animations[animation_name] = anim
            print(f"✅ Added '{animation_name}' to rig")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add animation to rig: {e}")
            return False
    
    def create_animation_file(self, name: str, output_path: str = None) -> str:
        """
        Create a template JSON file for a new animation.
        Edit this file and use add_custom() to add it.
        """
        template = {
            "category": "CUSTOM",
            "duration": 1.0,
            "loop": False,
            "description": f"Custom animation: {name}",
            "bones": {
                "torso": {
                    "rotate": [(0, 0), (0.5, 10), (1.0, 0)],
                    "translate": [(0, 0, 0), (0.5, 0, -10), (1.0, 0, 0)]
                },
                "head": {
                    "rotate": [(0, 0), (0.5, 5), (1.0, 0)]
                },
                "arm_left": {
                    "rotate": [(0, 0), (0.5, 20), (1.0, 0)]
                },
                "arm_right": {
                    "rotate": [(0, 0), (0.5, -20), (1.0, 0)]
                }
            }
        }
        
        if output_path is None:
            output_path = f"{name}.json"
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"✅ Created animation template: {output_path}")
        print("   Edit this file and use lib.add_custom() to add it!")
        
        return output_path
    
    def get_stats(self) -> Dict[str, int]:
        """Get library statistics"""
        stats = {"total": len(self.library)}
        
        for data in self.library.values():
            cat = data.get("category", "MISC")
            stats[cat] = stats.get(cat, 0) + 1
        
        return stats


# =============================================================================
# QUICK ACCESS FUNCTIONS
# =============================================================================

def list_animations():
    """Quick function to list all animations"""
    lib = AnimationLibrary()
    return lib.list_all()

def get_animation(name: str) -> Optional[Dict]:
    """Quick function to get an animation"""
    lib = AnimationLibrary()
    return lib.get(name)

def create_custom_animation(name: str, output_path: str = None) -> str:
    """Quick function to create a custom animation template"""
    lib = AnimationLibrary()
    return lib.create_animation_file(name, output_path)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    lib = AnimationLibrary()
    lib.list_all()
    
    print("\n" + "=" * 60)
    print("  USAGE EXAMPLES")
    print("=" * 60)
    print("""
    # List all animations:
    lib = AnimationLibrary()
    lib.list_all()
    
    # Get specific animation:
    anim = lib.get("attack_slash")
    
    # Search animations:
    results = lib.search("attack")
    
    # Add to a rig:
    lib.add_to_rig(my_rig, "jump")
    
    # Create custom animation:
    lib.create_animation_file("my_special_move")
    # Edit my_special_move.json, then:
    lib.add_custom("my_special_move.json")
    """)
