#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         SPINE RIG BUILDER                                    ║
║                                                                              ║
║   ONE-CLICK MONSTER IMAGE → GAME-READY SPINE RIG                            ║
║                                                                              ║
║   Usage:                                                                     ║
║   >>> builder = SpineRigBuilder()                                           ║
║   >>> builder.build("monster.png", archetype="humanoid")                    ║
║   >>> # Done! Full Spine rig with 20+ animations                            ║
║                                                                              ║
║   CLI:                                                                       ║
║   $ python spine_rig_builder.py monster.png --type humanoid --arms 4        ║
║                                                                              ║
║   Claude Code:                                                               ║
║   "Rig this demon with 6 arms and a cape for VEILBREAKERS"                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
import logging

# Local imports
from animation_engine import (
    PartType, CreatureArchetype, Bone, Slot, IKConstraint, PhysicsConstraint,
    Animation, CreatureRig, PartClassifier, BoneChainGenerator, AnimationGenerator,
    PHYSICS_PRESETS
)
from animation_templates import AnimationTemplates

# Try to import the rigger
try:
    from veilbreakers_rigger import VeilbreakersRigger, BODY_TEMPLATES, ExportFormat
    HAS_RIGGER = True
except ImportError:
    HAS_RIGGER = False

# Try to import precision segmenter
try:
    from precision_segmenter import PrecisionSegmenter, get_segmentation_status
    HAS_PRECISION_SEGMENTER = True
except ImportError:
    HAS_PRECISION_SEGMENTER = False

logger = logging.getLogger("SpineRigBuilder")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '\033[35m\033[1mBUILD\033[0m: \033[35m%(message)s\033[0m'
    ))
    logger.addHandler(handler)

# =============================================================================
# ARCHETYPE CONFIGURATIONS
# =============================================================================

ARCHETYPE_CONFIGS = {
    CreatureArchetype.HUMANOID: {
        "name": "Humanoid",
        "description": "Bipedal creatures (humans, demons, golems, knights)",
        "expected_parts": ["head", "torso", "arm_left", "arm_right", "leg_left", "leg_right"],
        "optional_parts": ["hair", "cape", "weapon", "shield", "tail", "wings"],
        "animations": [
            # Idle states
            "idle_breathe", "idle_combat", "idle",
            # Movement
            "walk", "run",
            # Basic attacks (matches game skill IDs)
            "attack_basic", "attack_slash", "attack_thrust", "attack_overhead",
            "rending_strike", "shield_bash", "execute",
            # Hit reactions
            "hit_light", "hit_heavy", "hit",
            # Death
            "death_fall_forward", "death_fall_backward", "death",
            # Special abilities
            "special_charge", "special_release", "frenzy", "apex_fury",
            # Defensive (matches game)
            "defend", "fortress_stance", "iron_wall", "cover_ally",
            # Utility
            "spawn", "victory", "taunt",
            # RPG cast actions (matches game skill types)
            "cast_heal", "siphon_heal", "life_tap", "essence_transfer",
            "cast_buff", "last_bastion",
            "cast_debuff", "fear_touch", "nightmare",
            "block", "dodge", "channel"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "torso"},
            "torso": {"type": PartType.RIGID_CORE, "parent": "root"},
            "arm_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "arm_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "leg_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "leg_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
        }
    },
    
    CreatureArchetype.MULTI_ARM: {
        "name": "Multi-Armed",
        "description": "Creatures with 4-10 arms (demons, spiders, eldritch)",
        "expected_parts": ["head", "torso"],
        "optional_parts": ["hair", "cape", "tentacle"],
        "arm_count_range": (4, 10),
        "animations": [
            "idle_breathe", "idle_menace",
            "attack_flurry", "attack_grab", "attack_slam",
            "hit_light", "hit_heavy",
            "death_fall_forward", "death_collapse",
            "special_charge", "special_frenzy",
            "spawn", "taunt",
            # RPG actions
            "cast_heal", "cast_buff", "cast_debuff", "channel"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "torso"},
            "torso": {"type": PartType.RIGID_CORE, "parent": "root"},
        }
    },
    
    CreatureArchetype.QUADRUPED: {
        "name": "Quadruped",
        "description": "Four-legged creatures (wolves, horses, dragons)",
        "expected_parts": ["head", "body", "leg_front_left", "leg_front_right", 
                          "leg_back_left", "leg_back_right"],
        "optional_parts": ["tail", "mane", "wings"],
        "animations": [
            "idle_breathe", "idle_combat", "walk", "run",
            "attack_bite", "attack_pounce", "attack_tail_sweep",
            "hit_light", "hit_heavy",
            "death_fall_forward",
            "special_roar", "special_charge",
            "spawn", "taunt"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "neck"},
            "neck": {"type": PartType.RIGID_CORE, "parent": "body"},
            "body": {"type": PartType.RIGID_CORE, "parent": "root"},
            "leg_front_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_front_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_back_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_back_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
        }
    },
    
    CreatureArchetype.SERPENT: {
        "name": "Serpent",
        "description": "Snake-like creatures (snakes, wyrms, lamia)",
        "expected_parts": ["head"],
        "body_segments": 6,
        "optional_parts": ["hood", "tongue", "tail_tip"],
        "animations": [
            "idle_breathe", "idle_menace", "slither",
            "attack_bite", "attack_constrict", "attack_spit",
            "hit_light", "hit_heavy",
            "death_fall_forward", "death_dissolve",
            "special_coil", "special_rise",
            "spawn"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "body_1"},
        }
    },
    
    CreatureArchetype.SKELETON: {
        "name": "Skeleton/Undead",
        "description": "Skeletal and undead creatures",
        "expected_parts": ["skull", "ribcage", "arm_left", "arm_right", 
                          "leg_left", "leg_right", "pelvis"],
        "optional_parts": ["jaw", "weapon", "shield", "cape"],
        "animations": [
            "idle_twitch", "shamble",
            "attack_slash", "attack_throw",
            "hit_light", "hit_heavy",
            "death_collapse",
            "spawn", "reassemble", "taunt"
        ],
        "bone_config": {
            "skull": {"type": PartType.SKELETAL, "parent": "ribcage"},
            "jaw": {"type": PartType.SKELETAL, "parent": "skull"},
            "ribcage": {"type": PartType.SKELETAL, "parent": "pelvis"},
            "pelvis": {"type": PartType.SKELETAL, "parent": "root"},
            "arm_left": {"type": PartType.SKELETAL, "parent": "ribcage", "bones": 3},
            "arm_right": {"type": PartType.SKELETAL, "parent": "ribcage", "bones": 3},
            "leg_left": {"type": PartType.SKELETAL, "parent": "pelvis", "bones": 3},
            "leg_right": {"type": PartType.SKELETAL, "parent": "pelvis", "bones": 3},
        }
    },
    
    CreatureArchetype.FLOATING: {
        "name": "Floating/Spectral",
        "description": "Floating creatures (ghosts, beholders, wisps)",
        "expected_parts": ["main_body"],
        "optional_parts": ["eye", "tentacle", "aura", "trail", "jaw", "horn"],
        "ethereal_trails": 4,  # Number of flowing trails
        "animations": [
            "idle_float", "idle_menace", "drift",
            "attack_beam", "attack_lunge", "attack_nova",
            "hit_light",
            "death_dissolve", "death_explode",
            "special_phase", "special_charge",
            "spawn", "taunt",
            # RPG actions
            "cast_heal", "cast_buff", "cast_debuff", "channel"
        ],
        "bone_config": {
            "main_body": {"type": PartType.FLOATING, "parent": "root"},
            "inner_core": {"type": PartType.FLOATING, "parent": "main_body"},
        }
    },
    
    CreatureArchetype.GIANT: {
        "name": "Giant",
        "description": "Large powerful creatures (titans, giants, colossi)",
        "expected_parts": ["head", "torso", "arm_left", "arm_right", 
                          "leg_left", "leg_right"],
        "optional_parts": ["hair", "armor", "weapon"],
        "animations": [
            "idle_breathe", "idle_menace", "walk",
            "attack_stomp", "attack_sweep", "attack_overhead",
            "hit_light", "hit_heavy",
            "death_fall_forward",
            "special_roar", "special_charge",
            "spawn"
        ],
        "animation_speed_mult": 0.7,  # Slower animations for size
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "torso"},
            "torso": {"type": PartType.RIGID_CORE, "parent": "root"},
            "arm_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "arm_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "leg_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "leg_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
        }
    },
    
    CreatureArchetype.INSECTOID: {
        "name": "Insectoid",
        "description": "Insect-like creatures (spiders, mantis, beetles)",
        "expected_parts": ["head", "thorax", "abdomen"],
        "leg_count": 6,
        "optional_parts": ["mandibles", "antenna", "wings", "stinger"],
        "animations": [
            "idle_twitch", "idle_clean", "scuttle",
            "attack_bite", "attack_sting", "attack_spit",
            "hit_light",
            "death_collapse",
            "special_molt", "special_swarm",
            "spawn"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "thorax"},
            "thorax": {"type": PartType.RIGID_CORE, "parent": "root"},
            "abdomen": {"type": PartType.RIGID_CORE, "parent": "thorax"},
        }
    },
    
    CreatureArchetype.WINGED: {
        "name": "Winged",
        "description": "Creatures with prominent wings (dragons, demons, angels)",
        "expected_parts": ["head", "body", "wing_left", "wing_right"],
        "optional_parts": ["tail", "arm_left", "arm_right", "leg_left", "leg_right"],
        "animations": [
            "idle_breathe", "idle_wings_fold",
            "walk", "fly", "hover",
            "attack_dive", "attack_claw", "attack_breath",
            "hit_light", "hit_heavy",
            "death_fall_forward",
            "special_roar", "special_take_flight",
            "spawn"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "body"},
            "body": {"type": PartType.RIGID_CORE, "parent": "root"},
            "wing_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 4},
            "wing_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 4},
        }
    },
    
    CreatureArchetype.ELDRITCH: {
        "name": "Eldritch/Cosmic Horror",
        "description": "Lovecraftian horrors with many appendages (The Congregation, The Weeping)",
        "expected_parts": ["main_body"],
        "tentacle_count": 8,
        "optional_parts": ["eye", "mouth", "wing", "head", "arm", "tentacle", "mass", "core", "jaw", "face_mass"],
        "animations": [
            # Idle states - writhing/pulsing
            "idle_writhe", "idle_pulse", "idle_menace", "idle",
            # Movement - floating/slithering
            "drift", "hover", "slither",
            # Attacks (matches game monster skills)
            "attack_grab", "attack_flurry", "attack_beam", "attack_slam",
            "collective_scream", "scream", "chorus", "chorus_of_names",
            "consume_the_weak", "assimilate", "absorb",
            "dread_gaze", "mass_confusion", "mass_drain",
            # Hit reactions
            "hit_light", "hit_heavy", "hit",
            # Death
            "death_dissolve", "death_explode", "death",
            # Special abilities
            "special_phase", "special_summon", "special_transform",
            "reality_shatter", "between_seconds",
            # Utility
            "spawn", "taunt",
            # RPG actions
            "cast_heal", "absorb_pain", "absorb_suffering",
            "cast_buff", "defined_purpose",
            "cast_debuff", "drown_in_despair", "nightmare",
            "channel"
        ],
        "bone_config": {
            "main_body": {"type": PartType.SOFT_TENTACLE, "parent": "root"},  # Writhing mass
        }
    },
    
    CreatureArchetype.AQUATIC: {
        "name": "Aquatic Creature",
        "description": "Fish, merfolk, sea monsters with fluid movement",
        "expected_parts": ["head", "body", "tail"],
        "optional_parts": ["fin_dorsal", "fin_left", "fin_right", "tentacle", "arm_left", "arm_right"],
        "animations": [
            "idle_drift", "idle_breathe",
            "swim_slow", "swim_fast", "swim_turn",
            "attack_bite", "attack_tail_whip", "attack_charge",
            "hit_light", "hit_heavy",
            "death_sink", "death_float",
            "special_dive", "special_surface",
            "spawn"
        ],
        "bone_config": {
            "body": {"type": PartType.RIGID_CORE, "parent": "root", "bones": 1},
            "head": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 1},
            "tail": {"type": PartType.SOFT_TENTACLE, "parent": "body", "bones": 5},
        },
        "tail_segments": 5,
        "physics_preset": "SOFT_BODY",
        "animation_speed_mult": 0.8,
    },

    # =========================================================================
    # HYBRID ARCHETYPES
    # =========================================================================

    CreatureArchetype.CENTAUR: {
        "name": "Centaur/Taur",
        "description": "Human upper body + quadruped lower body",
        "expected_parts": ["head", "torso", "arm_left", "arm_right", "body_lower",
                          "leg_front_left", "leg_front_right", "leg_back_left", "leg_back_right"],
        "optional_parts": ["tail", "hair", "weapon", "shield"],
        "animations": [
            "idle_breathe", "idle_combat", "idle",
            "walk", "run", "trot", "gallop", "rear_up",
            "attack_slash", "attack_thrust", "attack_trample", "attack_kick",
            "hit_light", "hit_heavy",
            "death_fall_forward", "death_collapse",
            "special_charge", "special_stomp",
            "spawn", "victory", "taunt",
            "cast_buff", "cast_heal", "block"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "torso"},
            "torso": {"type": PartType.RIGID_CORE, "parent": "body_lower"},
            "arm_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "arm_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "body_lower": {"type": PartType.RIGID_CORE, "parent": "root"},
            "leg_front_left": {"type": PartType.RIGID_LIMB, "parent": "body_lower", "bones": 3},
            "leg_front_right": {"type": PartType.RIGID_LIMB, "parent": "body_lower", "bones": 3},
            "leg_back_left": {"type": PartType.RIGID_LIMB, "parent": "body_lower", "bones": 3},
            "leg_back_right": {"type": PartType.RIGID_LIMB, "parent": "body_lower", "bones": 3},
        }
    },

    CreatureArchetype.NAGA: {
        "name": "Naga/Lamia",
        "description": "Human upper body + serpent lower body",
        "expected_parts": ["head", "torso", "arm_left", "arm_right", "tail"],
        "optional_parts": ["hair", "hood", "weapon", "crown"],
        "tail_segments": 8,
        "animations": [
            "idle_breathe", "idle_sway", "idle_menace", "idle",
            "slither", "coil", "strike",
            "attack_slash", "attack_bite", "attack_tail_whip", "attack_constrict",
            "hit_light", "hit_heavy",
            "death_collapse", "death",
            "special_hypnotize", "special_poison", "special_coil",
            "spawn", "taunt",
            "cast_heal", "cast_debuff", "cast_buff"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "torso"},
            "torso": {"type": PartType.RIGID_CORE, "parent": "tail"},
            "arm_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "arm_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "tail": {"type": PartType.SOFT_TENTACLE, "parent": "root", "bones": 8},
        }
    },

    CreatureArchetype.MERMAID: {
        "name": "Mermaid/Merfolk",
        "description": "Human upper body + fish tail",
        "expected_parts": ["head", "torso", "arm_left", "arm_right", "tail", "tail_fin"],
        "optional_parts": ["hair", "fins", "crown", "trident"],
        "tail_segments": 5,
        "animations": [
            "idle_breathe", "idle_float", "idle_swim", "idle",
            "swim", "swim_fast", "surface",
            "attack_scratch", "attack_tail_slap", "attack_trident",
            "hit_light", "hit_heavy",
            "death_sink", "death",
            "special_siren_song", "special_splash", "special_dive",
            "spawn", "taunt",
            "cast_heal", "cast_buff", "cast_debuff"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "torso"},
            "torso": {"type": PartType.RIGID_CORE, "parent": "tail"},
            "arm_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "arm_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "tail": {"type": PartType.SOFT_TENTACLE, "parent": "root", "bones": 5},
            "tail_fin": {"type": PartType.SOFT_HAIR, "parent": "tail", "bones": 2},
        }
    },

    CreatureArchetype.CHIMERA: {
        "name": "Chimera",
        "description": "Mixed body parts creature (lion+goat+snake, etc)",
        "expected_parts": ["head_main", "head_secondary", "body",
                          "leg_front_left", "leg_front_right", "leg_back_left", "leg_back_right", "tail"],
        "optional_parts": ["wings", "horns", "mane"],
        "animations": [
            "idle_breathe", "idle_heads_look", "idle",
            "walk", "run", "prowl",
            "attack_bite_main", "attack_bite_secondary", "attack_tail", "attack_pounce",
            "hit_light", "hit_heavy",
            "death_collapse", "death",
            "special_roar", "special_breath", "special_frenzy",
            "spawn", "taunt"
        ],
        "bone_config": {
            "head_main": {"type": PartType.RIGID_CORE, "parent": "body"},
            "head_secondary": {"type": PartType.RIGID_CORE, "parent": "body"},
            "body": {"type": PartType.RIGID_CORE, "parent": "root"},
            "leg_front_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_front_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_back_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_back_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "tail": {"type": PartType.SOFT_TENTACLE, "parent": "body", "bones": 5},
        }
    },

    # =========================================================================
    # CREATURE-SPECIFIC ARCHETYPES
    # =========================================================================

    CreatureArchetype.ARACHNID: {
        "name": "Arachnid/Spider",
        "description": "Eight-legged spider body",
        "expected_parts": ["head", "thorax", "abdomen"],
        "optional_parts": ["fangs", "pedipalps", "spinnerets"],
        "leg_count": 8,
        "animations": [
            "idle_breathe", "idle_twitch", "idle",
            "walk", "run", "crawl", "climb",
            "attack_bite", "attack_pounce", "attack_web_shoot",
            "hit_light", "hit_heavy",
            "death_curl", "death",
            "special_web_trap", "special_venom", "special_egg_sac",
            "spawn", "taunt"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "thorax"},
            "thorax": {"type": PartType.RIGID_CORE, "parent": "root"},
            "abdomen": {"type": PartType.RIGID_CORE, "parent": "thorax"},
        }
    },

    CreatureArchetype.AVIAN: {
        "name": "Avian/Bird",
        "description": "Bird anatomy with wings as primary limbs",
        "expected_parts": ["head", "body", "wing_left", "wing_right", "leg_left", "leg_right", "tail_feathers"],
        "optional_parts": ["crest", "beak"],
        "animations": [
            "idle_breathe", "idle_preen", "idle_look", "idle",
            "walk", "hop", "fly", "glide", "land", "take_off",
            "attack_peck", "attack_claw", "attack_dive", "attack_wing_buffet",
            "hit_light", "hit_heavy",
            "death_fall", "death",
            "special_screech", "special_feather_storm",
            "spawn", "taunt"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "body"},
            "body": {"type": PartType.RIGID_CORE, "parent": "root"},
            "wing_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 4},
            "wing_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 4},
            "leg_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "tail_feathers": {"type": PartType.SOFT_HAIR, "parent": "body", "bones": 3},
        }
    },

    CreatureArchetype.AMPHIBIAN: {
        "name": "Amphibian",
        "description": "Frog, salamander, newt body type",
        "expected_parts": ["head", "body", "leg_front_left", "leg_front_right",
                          "leg_back_left", "leg_back_right"],
        "optional_parts": ["tail", "tongue", "throat_sac", "crest"],
        "animations": [
            "idle_breathe", "idle_croak", "idle",
            "walk", "hop", "swim", "climb",
            "attack_tongue", "attack_bite", "attack_leap",
            "hit_light", "hit_heavy",
            "death_flop", "death",
            "special_croak", "special_poison_skin", "special_camouflage",
            "spawn", "taunt"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "body"},
            "body": {"type": PartType.RIGID_CORE, "parent": "root"},
            "leg_front_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_front_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_back_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_back_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
        }
    },

    CreatureArchetype.CRUSTACEAN: {
        "name": "Crustacean",
        "description": "Crab, lobster, shrimp with shell and claws",
        "expected_parts": ["head", "body", "claw_left", "claw_right",
                          "leg_1", "leg_2", "leg_3", "leg_4", "leg_5", "leg_6"],
        "optional_parts": ["antennae", "tail", "shell_spikes"],
        "animations": [
            "idle_breathe", "idle_bubble", "idle",
            "walk_sideways", "walk_forward", "burrow",
            "attack_pinch", "attack_claw_slam", "attack_charge",
            "hit_light", "hit_heavy",
            "death_flip", "death",
            "special_shell_defense", "special_molt",
            "spawn", "taunt"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "body"},
            "body": {"type": PartType.RIGID_CORE, "parent": "root"},
            "claw_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "claw_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
        }
    },

    CreatureArchetype.WORM: {
        "name": "Worm/Centipede",
        "description": "Many-segmented crawler",
        "expected_parts": ["head", "segment_1", "segment_2", "segment_3", "segment_4",
                          "segment_5", "segment_6", "segment_7", "segment_8", "tail"],
        "optional_parts": ["mandibles", "antennae", "legs"],
        "segment_count": 10,
        "animations": [
            "idle_writhe", "idle_coil", "idle",
            "crawl", "burrow", "surface",
            "attack_bite", "attack_coil", "attack_spit",
            "hit_light", "hit_heavy",
            "death_writhe", "death",
            "special_split", "special_regenerate", "special_acid",
            "spawn"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "segment_1"},
        }
    },

    CreatureArchetype.DRAGON: {
        "name": "Dragon",
        "description": "Full dragon with four legs, wings, and tail",
        "expected_parts": ["head", "neck", "body", "wing_left", "wing_right",
                          "leg_front_left", "leg_front_right", "leg_back_left", "leg_back_right", "tail"],
        "optional_parts": ["horns", "spines", "jaw"],
        "animations": [
            "idle_breathe", "idle_menace", "idle_sleep", "idle",
            "walk", "run", "fly", "glide", "land", "take_off", "hover",
            "attack_bite", "attack_claw", "attack_tail_sweep", "attack_wing_buffet",
            "attack_breath_fire", "attack_breath_ice", "attack_breath_lightning",
            "hit_light", "hit_heavy",
            "death_crash", "death",
            "special_roar", "special_intimidate", "special_stomp",
            "spawn", "victory", "taunt"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "neck"},
            "neck": {"type": PartType.SOFT_TENTACLE, "parent": "body", "bones": 4},
            "body": {"type": PartType.RIGID_CORE, "parent": "root"},
            "wing_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 5},
            "wing_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 5},
            "leg_front_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_front_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_back_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_back_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "tail": {"type": PartType.SOFT_TENTACLE, "parent": "body", "bones": 6},
        }
    },

    # =========================================================================
    # FANTASY/MYTHICAL ARCHETYPES
    # =========================================================================

    CreatureArchetype.DEMON: {
        "name": "Demon",
        "description": "Demonic creature with horns, wings, and tail",
        "expected_parts": ["head", "torso", "arm_left", "arm_right", "leg_left", "leg_right"],
        "optional_parts": ["horns", "wings", "tail", "cloven_hooves"],
        "animations": [
            "idle_breathe", "idle_menace", "idle_flame", "idle",
            "walk", "run", "fly", "hover",
            "attack_slash", "attack_fireball", "attack_grab", "attack_tail_whip",
            "hit_light", "hit_heavy",
            "death_banish", "death",
            "special_summon", "special_hellfire", "special_corrupt", "special_possess",
            "spawn", "taunt", "laugh",
            "cast_debuff", "cast_buff", "channel"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "torso"},
            "torso": {"type": PartType.RIGID_CORE, "parent": "root"},
            "arm_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "arm_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "leg_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "leg_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
        }
    },

    CreatureArchetype.ANGEL: {
        "name": "Angel",
        "description": "Celestial being with wings and halo",
        "expected_parts": ["head", "torso", "arm_left", "arm_right", "leg_left", "leg_right",
                          "wing_left", "wing_right"],
        "optional_parts": ["halo", "sword", "shield", "robes"],
        "animations": [
            "idle_breathe", "idle_pray", "idle_glow", "idle",
            "walk", "fly", "hover", "glide", "descend", "ascend",
            "attack_smite", "attack_sword", "attack_light_beam",
            "hit_light", "hit_heavy",
            "death_ascend", "death",
            "special_blessing", "special_holy_light", "special_resurrect",
            "spawn", "victory",
            "cast_heal", "cast_buff", "channel"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "torso"},
            "torso": {"type": PartType.RIGID_CORE, "parent": "root"},
            "arm_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "arm_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "leg_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "leg_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "wing_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 4},
            "wing_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 4},
        }
    },

    CreatureArchetype.UNDEAD: {
        "name": "Undead",
        "description": "Zombie, lich, ghoul, revenant",
        "expected_parts": ["head", "torso", "arm_left", "arm_right", "leg_left", "leg_right"],
        "optional_parts": ["jaw", "ribs", "entrails", "weapon"],
        "animations": [
            "idle_sway", "idle_moan", "idle",
            "shamble", "lurch", "crawl",
            "attack_bite", "attack_claw", "attack_grab",
            "hit_light", "hit_heavy", "hit_limb_off",
            "death_collapse", "death_crumble", "death",
            "special_rise", "special_infect", "special_scream",
            "spawn"
        ],
        "bone_config": {
            "head": {"type": PartType.SKELETAL, "parent": "torso"},
            "torso": {"type": PartType.SKELETAL, "parent": "root"},
            "arm_left": {"type": PartType.SKELETAL, "parent": "torso", "bones": 3},
            "arm_right": {"type": PartType.SKELETAL, "parent": "torso", "bones": 3},
            "leg_left": {"type": PartType.SKELETAL, "parent": "torso", "bones": 3},
            "leg_right": {"type": PartType.SKELETAL, "parent": "torso", "bones": 3},
        }
    },

    CreatureArchetype.SPECTRAL: {
        "name": "Spectral/Ghost",
        "description": "Ghost, wraith, phantom (transparent, floaty)",
        "expected_parts": ["head", "body", "arm_left", "arm_right"],
        "optional_parts": ["trail", "chains", "lantern"],
        "animations": [
            "idle_float", "idle_flicker", "idle_wail", "idle",
            "drift", "phase", "vanish", "appear",
            "attack_chill_touch", "attack_possess", "attack_scream",
            "hit_light", "hit_disperse",
            "death_disperse", "death",
            "special_haunt", "special_curse", "special_phase_through",
            "spawn"
        ],
        "bone_config": {
            "head": {"type": PartType.FLOATING, "parent": "body"},
            "body": {"type": PartType.FLOATING, "parent": "root"},
            "arm_left": {"type": PartType.SOFT_HAIR, "parent": "body", "bones": 3},
            "arm_right": {"type": PartType.SOFT_HAIR, "parent": "body", "bones": 3},
        }
    },

    CreatureArchetype.ELEMENTAL: {
        "name": "Elemental",
        "description": "Fire, water, earth, or air elemental being",
        "expected_parts": ["core", "body"],
        "optional_parts": ["arm_left", "arm_right", "aura", "particles"],
        "animations": [
            "idle_pulse", "idle_swirl", "idle_crackle", "idle",
            "move", "surge", "dissipate", "reform",
            "attack_blast", "attack_wave", "attack_slam",
            "hit_light", "hit_scatter",
            "death_disperse", "death",
            "special_elemental_fury", "special_absorb", "special_transform",
            "spawn"
        ],
        "bone_config": {
            "core": {"type": PartType.FLOATING, "parent": "body"},
            "body": {"type": PartType.FLOATING, "parent": "root"},
        }
    },

    # =========================================================================
    # ABERRANT ARCHETYPES
    # =========================================================================

    CreatureArchetype.SLIME: {
        "name": "Slime/Ooze",
        "description": "Formless blob with no skeleton",
        "expected_parts": ["body"],
        "optional_parts": ["core", "eye", "pseudopod"],
        "animations": [
            "idle_wobble", "idle_bubble", "idle_pulse", "idle",
            "ooze", "bounce", "split", "merge",
            "attack_engulf", "attack_spit", "attack_pseudopod",
            "hit_wobble", "hit_splash",
            "death_splat", "death_evaporate", "death",
            "special_divide", "special_absorb", "special_acid_pool",
            "spawn"
        ],
        "bone_config": {
            "body": {"type": PartType.FLOATING, "parent": "root"},
        }
    },

    CreatureArchetype.TENTACLE_BEAST: {
        "name": "Tentacle Beast",
        "description": "Primarily tentacles (octopus, kraken, mind flayer)",
        "expected_parts": ["head", "body"],
        "optional_parts": ["beak", "eye"],
        "tentacle_count": 8,
        "animations": [
            "idle_writhe", "idle_pulse", "idle",
            "crawl", "swim", "jet",
            "attack_grab", "attack_slam", "attack_constrict", "attack_ink",
            "hit_light", "hit_heavy",
            "death_collapse", "death",
            "special_hypnotize", "special_ink_cloud", "special_regenerate",
            "spawn"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "body"},
            "body": {"type": PartType.RIGID_CORE, "parent": "root"},
        }
    },

    CreatureArchetype.MULTI_HEAD: {
        "name": "Multi-Headed",
        "description": "Hydra, Cerberus, Ettin (multiple heads)",
        "expected_parts": ["head_1", "head_2", "head_3", "body",
                          "leg_front_left", "leg_front_right", "leg_back_left", "leg_back_right"],
        "optional_parts": ["head_4", "head_5", "tail", "wings"],
        "head_count": 3,
        "animations": [
            "idle_breathe", "idle_heads_argue", "idle_heads_sync", "idle",
            "walk", "run",
            "attack_bite_1", "attack_bite_2", "attack_bite_3", "attack_bite_all",
            "attack_breath", "attack_pounce",
            "hit_light", "hit_heavy", "hit_head_severed",
            "death_collapse", "death",
            "special_head_regrow", "special_synchronized_roar",
            "spawn", "taunt"
        ],
        "bone_config": {
            "head_1": {"type": PartType.RIGID_CORE, "parent": "body"},
            "head_2": {"type": PartType.RIGID_CORE, "parent": "body"},
            "head_3": {"type": PartType.RIGID_CORE, "parent": "body"},
            "body": {"type": PartType.RIGID_CORE, "parent": "root"},
            "leg_front_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_front_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_back_left": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
            "leg_back_right": {"type": PartType.RIGID_LIMB, "parent": "body", "bones": 3},
        }
    },

    CreatureArchetype.CYCLOPS: {
        "name": "Cyclops",
        "description": "One-eyed giant",
        "expected_parts": ["head", "eye", "torso", "arm_left", "arm_right", "leg_left", "leg_right"],
        "optional_parts": ["club", "armor", "beard"],
        "animations": [
            "idle_breathe", "idle_look", "idle",
            "walk", "stomp",
            "attack_club_smash", "attack_throw", "attack_grab", "attack_kick",
            "hit_light", "hit_heavy", "hit_eye",
            "death_fall", "death",
            "special_roar", "special_ground_pound", "special_boulder_throw",
            "spawn", "taunt"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "torso"},
            "eye": {"type": PartType.RIGID_CORE, "parent": "head"},
            "torso": {"type": PartType.RIGID_CORE, "parent": "root"},
            "arm_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "arm_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "leg_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "leg_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
        },
        "animation_speed_mult": 0.7
    },

    CreatureArchetype.SWARM: {
        "name": "Swarm",
        "description": "Colony of small creatures (bees, rats, bats)",
        "expected_parts": ["swarm_core"],
        "optional_parts": ["outliers"],
        "swarm_count": 20,
        "animations": [
            "idle_buzz", "idle_swirl", "idle",
            "move", "scatter", "reform", "expand", "contract",
            "attack_engulf", "attack_sting", "attack_bite_swarm",
            "hit_disperse", "hit_reform",
            "death_scatter", "death",
            "special_frenzy", "special_infest",
            "spawn"
        ],
        "bone_config": {
            "swarm_core": {"type": PartType.FLOATING, "parent": "root"},
        }
    },

    # =========================================================================
    # CONSTRUCT/OTHER ARCHETYPES
    # =========================================================================

    CreatureArchetype.MECHANICAL: {
        "name": "Mechanical/Construct",
        "description": "Robot, golem, clockwork creature",
        "expected_parts": ["head", "torso", "arm_left", "arm_right", "leg_left", "leg_right"],
        "optional_parts": ["gears", "steam", "weapons", "shield"],
        "animations": [
            "idle_hum", "idle_scan", "idle",
            "walk", "run", "stomp",
            "attack_punch", "attack_laser", "attack_missiles", "attack_spin",
            "hit_light", "hit_spark", "hit_malfunction",
            "death_explode", "death_shutdown", "death",
            "special_transform", "special_overcharge", "special_repair",
            "spawn", "activate", "deactivate"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "torso"},
            "torso": {"type": PartType.RIGID_CORE, "parent": "root"},
            "arm_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "arm_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "leg_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
            "leg_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 3},
        }
    },

    CreatureArchetype.CRYSTALLINE: {
        "name": "Crystalline",
        "description": "Crystal or gem creature",
        "expected_parts": ["core", "crystal_1", "crystal_2", "crystal_3"],
        "optional_parts": ["crystal_4", "crystal_5", "aura"],
        "animations": [
            "idle_pulse", "idle_shimmer", "idle",
            "float", "glide",
            "attack_shard", "attack_beam", "attack_shatter",
            "hit_crack", "hit_chip",
            "death_shatter", "death",
            "special_reflect", "special_grow", "special_refract",
            "spawn"
        ],
        "bone_config": {
            "core": {"type": PartType.FLOATING, "parent": "root"},
            "crystal_1": {"type": PartType.RIGID_CORE, "parent": "core"},
            "crystal_2": {"type": PartType.RIGID_CORE, "parent": "core"},
            "crystal_3": {"type": PartType.RIGID_CORE, "parent": "core"},
        }
    },

    CreatureArchetype.FUNGAL: {
        "name": "Fungal/Mushroom",
        "description": "Mushroom creature, myconid",
        "expected_parts": ["cap", "stem", "arm_left", "arm_right"],
        "optional_parts": ["spores", "tendrils", "smaller_caps"],
        "animations": [
            "idle_breathe", "idle_sway", "idle_spore", "idle",
            "waddle", "root", "uproot",
            "attack_spore_cloud", "attack_slam", "attack_tendril",
            "hit_light", "hit_puff",
            "death_collapse", "death_spore_burst", "death",
            "special_infect", "special_spread", "special_heal_aura",
            "spawn"
        ],
        "bone_config": {
            "cap": {"type": PartType.RIGID_CORE, "parent": "stem"},
            "stem": {"type": PartType.RIGID_CORE, "parent": "root"},
            "arm_left": {"type": PartType.SOFT_TENTACLE, "parent": "stem", "bones": 3},
            "arm_right": {"type": PartType.SOFT_TENTACLE, "parent": "stem", "bones": 3},
        }
    },

    CreatureArchetype.PLANT: {
        "name": "Plant/Treant",
        "description": "Treant, vine creature, plant monster",
        "expected_parts": ["head", "trunk", "branch_left", "branch_right", "root_left", "root_right"],
        "optional_parts": ["leaves", "flowers", "vines", "fruit"],
        "animations": [
            "idle_sway", "idle_rustle", "idle",
            "walk", "root", "uproot",
            "attack_branch_slam", "attack_vine_whip", "attack_root_grab",
            "hit_light", "hit_chip",
            "death_wilt", "death_topple", "death",
            "special_entangle", "special_bloom", "special_thorns",
            "spawn"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "trunk"},
            "trunk": {"type": PartType.RIGID_CORE, "parent": "root"},
            "branch_left": {"type": PartType.RIGID_LIMB, "parent": "trunk", "bones": 4},
            "branch_right": {"type": PartType.RIGID_LIMB, "parent": "trunk", "bones": 4},
            "root_left": {"type": PartType.SOFT_TENTACLE, "parent": "trunk", "bones": 3},
            "root_right": {"type": PartType.SOFT_TENTACLE, "parent": "trunk", "bones": 3},
        }
    },

    CreatureArchetype.SHADOW: {
        "name": "Shadow",
        "description": "Shadow being, living darkness",
        "expected_parts": ["body"],
        "optional_parts": ["tendrils", "eyes", "claws"],
        "animations": [
            "idle_flicker", "idle_undulate", "idle",
            "glide", "sink", "rise", "teleport",
            "attack_claw", "attack_engulf", "attack_drain",
            "hit_disperse", "hit_reform",
            "death_fade", "death",
            "special_possess", "special_darkness", "special_fear",
            "spawn"
        ],
        "bone_config": {
            "body": {"type": PartType.FLOATING, "parent": "root"},
        }
    },

    CreatureArchetype.COLOSSUS: {
        "name": "Colossus",
        "description": "Massive scale creature (building-sized)",
        "expected_parts": ["head", "torso", "arm_left", "arm_right", "leg_left", "leg_right"],
        "optional_parts": ["armor_plates", "weapon", "crystals"],
        "animations": [
            "idle_breathe", "idle_rumble", "idle",
            "walk", "stomp",
            "attack_slam", "attack_sweep", "attack_stomp", "attack_beam",
            "hit_light", "hit_heavy", "hit_stagger",
            "death_collapse", "death_crumble", "death",
            "special_roar", "special_earthquake", "special_summon_minions",
            "spawn", "rise"
        ],
        "bone_config": {
            "head": {"type": PartType.RIGID_CORE, "parent": "torso"},
            "torso": {"type": PartType.RIGID_CORE, "parent": "root"},
            "arm_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 4},
            "arm_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 4},
            "leg_left": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 4},
            "leg_right": {"type": PartType.RIGID_LIMB, "parent": "torso", "bones": 4},
        },
        "animation_speed_mult": 0.5
    },

    CreatureArchetype.PARASITIC: {
        "name": "Parasitic",
        "description": "Attaches to host creature",
        "expected_parts": ["body", "attachment_point"],
        "optional_parts": ["tendrils", "mouth", "eye"],
        "animations": [
            "idle_pulse", "idle_feed", "idle",
            "crawl", "attach", "detach",
            "attack_drain", "attack_control", "attack_infect",
            "hit_light", "hit_detach",
            "death_shrivel", "death",
            "special_mind_control", "special_spread", "special_burst",
            "spawn"
        ],
        "bone_config": {
            "body": {"type": PartType.SOFT_TENTACLE, "parent": "attachment_point"},
            "attachment_point": {"type": PartType.RIGID_CORE, "parent": "root"},
        }
    },

    CreatureArchetype.MIMIC: {
        "name": "Mimic",
        "description": "Shapeshifter base form",
        "expected_parts": ["body", "mouth"],
        "optional_parts": ["tongue", "teeth", "eyes", "pseudopods"],
        "animations": [
            "idle_still", "idle_subtle_move", "idle",
            "transform_in", "transform_out",
            "attack_bite", "attack_tongue", "attack_pseudopod",
            "hit_light", "hit_reveal",
            "death_melt", "death",
            "special_disguise", "special_ambush", "special_adapt",
            "spawn"
        ],
        "bone_config": {
            "body": {"type": PartType.FLOATING, "parent": "root"},
            "mouth": {"type": PartType.RIGID_CORE, "parent": "body"},
        }
    },

    # =========================================================================
    # FALLBACK
    # =========================================================================

    CreatureArchetype.CUSTOM: {
        "name": "Custom Creature",
        "description": "User-defined creature with custom parts",
        "expected_parts": ["body"],
        "optional_parts": [],
        "animations": [
            "idle_breathe",
            "walk",
            "attack",
            "hit",
            "death",
            "spawn"
        ],
        "bone_config": {
            "body": {"type": PartType.RIGID_CORE, "parent": "root", "bones": 1},
        }
    },
}

# =============================================================================
# SPINE RIG BUILDER
# =============================================================================

class SpineRigBuilder:
    """
    The Ultimate One-Click Monster Rig Builder
    
    Takes a monster image and outputs a complete Spine rig with:
    - Auto-detected and classified body parts
    - Proper bone hierarchy with IK
    - Physics constraints for soft parts (hair, cape, tentacles)
    - 15-25 pre-built animations appropriate for creature type
    - Game-ready export in Spine JSON format
    """
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.classifier = PartClassifier()
        self.chain_generator = BoneChainGenerator(self.classifier)
        
        # Initialize precision segmenter if available
        if HAS_PRECISION_SEGMENTER:
            self.precision_segmenter = PrecisionSegmenter()
            seg_status = get_segmentation_status()
            if seg_status.get("sam2"):
                logger.info("SAM2 precision segmentation available (99.9% accuracy)")
            elif seg_status.get("opencv"):
                logger.info("OpenCV segmentation available (85% accuracy)")
            else:
                logger.info("Basic segmentation available")
        else:
            self.precision_segmenter = None
        
        # Initialize rigger if available
        if HAS_RIGGER:
            self.rigger = VeilbreakersRigger(
                output_dir=str(self.output_dir),
                use_fallback=True
            )
        else:
            self.rigger = None
            logger.warning("VeilbreakersRigger not available - using manual part input")
        
        logger.info("SpineRigBuilder initialized ✓")
    
    def build(self,
              image_path: str,
              name: Optional[str] = None,
              archetype: str = "humanoid",
              arm_count: int = 2,
              leg_count: int = 2,
              has_tail: bool = False,
              has_wings: bool = False,
              has_hair: bool = False,
              has_cape: bool = False,
              tentacle_count: int = 0,
              custom_parts: Optional[Dict[str, dict]] = None,
              animation_speed: float = 1.0) -> str:
        """
        Build a complete Spine rig from an image
        
        Args:
            image_path: Path to monster image
            name: Rig name (defaults to filename)
            archetype: Creature type (humanoid, quadruped, etc.)
            arm_count: Number of arms (for multi-arm types)
            leg_count: Number of legs
            has_tail: Include tail with physics
            has_wings: Include wings
            has_hair: Include hair with physics
            has_cape: Include cape with physics
            tentacle_count: Number of tentacles
            custom_parts: Dict of custom part definitions
            animation_speed: Speed multiplier for animations
            
        Returns:
            Path to exported Spine JSON file
        """
        # Parse archetype
        arch_enum = self._parse_archetype(archetype)
        config = ARCHETYPE_CONFIGS.get(arch_enum, ARCHETYPE_CONFIGS[CreatureArchetype.HUMANOID])
        
        # Get name from image if not provided
        if name is None:
            name = Path(image_path).stem
        
        logger.info(f"Building rig: {name} ({config['name']})")
        
        # Load and process image
        image = Image.open(image_path).convert("RGBA")
        width, height = image.size
        
        # Create rig structure
        rig = CreatureRig(
            name=name,
            archetype=arch_enum,
            width=width,
            height=height
        )
        
        # Add root bone
        rig.bones.append(Bone(
            name="root",
            x=width / 2,
            y=height * 0.1,  # Near bottom
            length=0,
            color="00FF00FF"
        ))
        
        # Segment image and build skeleton
        parts_data = self._segment_image(image_path, name, config)
        
        # Build bones for each part
        self._build_skeleton(rig, parts_data, config, arm_count, leg_count,
                            has_tail, has_wings, has_hair, has_cape, tentacle_count)
        
        # Add custom parts if provided
        if custom_parts:
            self._add_custom_parts(rig, custom_parts)
        
        # Create slots for all parts
        self._create_slots(rig, parts_data)
        
        # Generate animations
        speed_mult = config.get("animation_speed_mult", 1.0) * animation_speed
        self._generate_animations(rig, config, speed_mult)
        
        # Add standard events
        self._add_events(rig)
        
        # Export
        output_path = self._export(rig, image_path)

        logger.info(f"✅ Rig complete: {output_path}")
        logger.info(f"   Bones: {len(rig.bones)}")
        logger.info(f"   Animations: {len(rig.animations)}")

        return output_path

    def build_rig(self,
                  parts: list,
                  archetype: str = "humanoid",
                  rig_name: str = "monster",
                  arm_count: int = 2,
                  leg_count: int = 2,
                  has_tail: bool = False,
                  has_wings: bool = False,
                  has_hair: bool = False,
                  has_cape: bool = False,
                  tentacle_count: int = 0,
                  animation_speed: float = 1.0) -> dict:
        """
        Build a Spine rig from pre-detected parts (called from UI).

        Args:
            parts: List of BodyPart objects from the rigger
            archetype: Creature type
            rig_name: Name for the rig
            arm_count: Number of arms
            leg_count: Number of legs
            has_tail: Include tail
            has_wings: Include wings
            has_hair: Include hair physics
            has_cape: Include cape physics
            tentacle_count: Number of tentacles
            animation_speed: Speed multiplier

        Returns:
            Spine JSON data as dict (ready to save)
        """
        # Parse archetype
        arch_enum = self._parse_archetype(archetype)
        config = ARCHETYPE_CONFIGS.get(arch_enum, ARCHETYPE_CONFIGS[CreatureArchetype.HUMANOID])

        logger.info(f"Building rig from {len(parts)} parts: {rig_name} ({config['name']})")

        # Get dimensions from parts
        width, height = 512, 512  # Default
        for part in parts:
            if hasattr(part, 'image') and part.image is not None:
                h, w = part.image.shape[:2]
                width = max(width, w)
                height = max(height, h)
            elif hasattr(part, 'mask') and part.mask is not None:
                h, w = part.mask.shape[:2]
                width = max(width, w)
                height = max(height, h)

        # Create rig structure
        rig = CreatureRig(
            name=rig_name,
            archetype=arch_enum,
            width=width,
            height=height
        )

        # Add root bone at center-bottom
        rig.bones.append(Bone(
            name="root",
            x=width / 2,
            y=height * 0.1,
            length=0,
            color="00FF00FF"
        ))

        # Convert parts to parts_data format
        parts_data = {}
        for part in parts:
            part_name = part.name.lower()

            # Get bounding box
            bbox = None
            if hasattr(part, 'bbox') and part.bbox:
                bbox = (part.bbox.x1, part.bbox.y1, part.bbox.x2, part.bbox.y2)
            elif hasattr(part, 'bounds') and part.bounds:
                bbox = part.bounds
            elif hasattr(part, 'mask') and part.mask is not None:
                ys, xs = np.where(part.mask > 0)
                if len(xs) > 0:
                    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

            if bbox:
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                parts_data[part_name] = {
                    "name": part_name,
                    "bbox": bbox,
                    "center": (cx, cy),
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "z_index": getattr(part, 'z_index', 0)
                }

        # Build skeleton
        self._build_skeleton(rig, parts_data, config, arm_count, leg_count,
                            has_tail, has_wings, has_hair, has_cape, tentacle_count)

        # Create slots
        self._create_slots(rig, parts_data)

        # Generate animations
        speed_mult = config.get("animation_speed_mult", 1.0) * animation_speed
        self._generate_animations(rig, config, speed_mult)

        # Add events
        self._add_events(rig)

        # Convert to Spine JSON format
        spine_data = self._to_spine_json(rig)

        logger.info(f"✅ Rig built: {len(rig.bones)} bones, {len(rig.animations)} animations")

        return spine_data

    def _to_spine_json(self, rig: 'CreatureRig') -> dict:
        """Convert CreatureRig to Spine JSON format"""
        # Build bones array
        bones = []
        for bone in rig.bones:
            bone_data = {
                "name": bone.name,
                "x": bone.x,
                "y": bone.y,
                "length": bone.length,
                "rotation": bone.rotation
            }
            if bone.parent:
                bone_data["parent"] = bone.parent
            if bone.color:
                bone_data["color"] = bone.color
            bones.append(bone_data)

        # Build slots array
        slots = []
        for slot in rig.slots:
            slot_data = {
                "name": slot.name,
                "bone": slot.bone,
                "attachment": slot.attachment
            }
            slots.append(slot_data)

        # Build skins
        skin_attachments = {}
        for slot in rig.slots:
            if slot.attachment:
                skin_attachments[slot.name] = {
                    slot.attachment: {
                        "type": "region",
                        "width": 100,
                        "height": 100
                    }
                }

        # Build animations
        animations = {}
        for anim in rig.animations:
            anim_data = {"bones": {}}
            for bone_name, keyframes in anim.bone_timelines.items():
                bone_timeline = {}
                if "rotate" in keyframes:
                    bone_timeline["rotate"] = keyframes["rotate"]
                if "translate" in keyframes:
                    bone_timeline["translate"] = keyframes["translate"]
                if "scale" in keyframes:
                    bone_timeline["scale"] = keyframes["scale"]
                if bone_timeline:
                    anim_data["bones"][bone_name] = bone_timeline
            animations[anim.name] = anim_data

        return {
            "skeleton": {
                "hash": rig.name,
                "spine": "4.1",
                "width": rig.width,
                "height": rig.height
            },
            "bones": bones,
            "slots": slots,
            "skins": [{"name": "default", "attachments": skin_attachments}],
            "animations": animations
        }

    def _parse_archetype(self, archetype: str) -> CreatureArchetype:
        """Parse archetype string to enum - supports all 39 creature types"""
        arch_map = {
            # === STANDARD ARCHETYPES ===
            "humanoid": CreatureArchetype.HUMANOID,
            "human": CreatureArchetype.HUMANOID,
            "bipedal": CreatureArchetype.HUMANOID,
            "person": CreatureArchetype.HUMANOID,

            "multi_arm": CreatureArchetype.MULTI_ARM,
            "multiarm": CreatureArchetype.MULTI_ARM,
            "multi-arm": CreatureArchetype.MULTI_ARM,
            "many_arms": CreatureArchetype.MULTI_ARM,

            "quadruped": CreatureArchetype.QUADRUPED,
            "quad": CreatureArchetype.QUADRUPED,
            "fourleg": CreatureArchetype.QUADRUPED,
            "four_legs": CreatureArchetype.QUADRUPED,
            "wolf": CreatureArchetype.QUADRUPED,
            "horse": CreatureArchetype.QUADRUPED,
            "dog": CreatureArchetype.QUADRUPED,
            "cat": CreatureArchetype.QUADRUPED,
            "bear": CreatureArchetype.QUADRUPED,
            "lion": CreatureArchetype.QUADRUPED,

            "serpent": CreatureArchetype.SERPENT,
            "snake": CreatureArchetype.SERPENT,
            "wyrm": CreatureArchetype.SERPENT,
            "eel": CreatureArchetype.SERPENT,

            "skeleton": CreatureArchetype.SKELETON,
            "bones": CreatureArchetype.SKELETON,
            "skeletal": CreatureArchetype.SKELETON,

            "floating": CreatureArchetype.FLOATING,
            "wisp": CreatureArchetype.FLOATING,
            "orb": CreatureArchetype.FLOATING,
            "eye": CreatureArchetype.FLOATING,

            "giant": CreatureArchetype.GIANT,
            "titan": CreatureArchetype.GIANT,
            "ogre": CreatureArchetype.GIANT,
            "troll": CreatureArchetype.GIANT,

            "insectoid": CreatureArchetype.INSECTOID,
            "insect": CreatureArchetype.INSECTOID,
            "beetle": CreatureArchetype.INSECTOID,
            "mantis": CreatureArchetype.INSECTOID,
            "ant": CreatureArchetype.INSECTOID,

            "winged": CreatureArchetype.WINGED,
            "bat": CreatureArchetype.WINGED,
            "wyvern": CreatureArchetype.WINGED,

            "aquatic": CreatureArchetype.AQUATIC,
            "fish": CreatureArchetype.AQUATIC,
            "shark": CreatureArchetype.AQUATIC,
            "sea": CreatureArchetype.AQUATIC,

            "eldritch": CreatureArchetype.ELDRITCH,
            "cosmic": CreatureArchetype.ELDRITCH,
            "lovecraft": CreatureArchetype.ELDRITCH,
            "aberration": CreatureArchetype.ELDRITCH,
            "horror": CreatureArchetype.ELDRITCH,

            # === HYBRID ARCHETYPES ===
            "centaur": CreatureArchetype.CENTAUR,
            "horseman": CreatureArchetype.CENTAUR,

            "naga": CreatureArchetype.NAGA,
            "lamia": CreatureArchetype.NAGA,
            "medusa": CreatureArchetype.NAGA,
            "yuan-ti": CreatureArchetype.NAGA,

            "mermaid": CreatureArchetype.MERMAID,
            "merman": CreatureArchetype.MERMAID,
            "siren": CreatureArchetype.MERMAID,
            "merfolk": CreatureArchetype.MERMAID,

            "chimera": CreatureArchetype.CHIMERA,
            "hybrid": CreatureArchetype.CHIMERA,
            "mixed": CreatureArchetype.CHIMERA,
            "manticore": CreatureArchetype.CHIMERA,

            # === CREATURE-SPECIFIC ===
            "arachnid": CreatureArchetype.ARACHNID,
            "spider": CreatureArchetype.ARACHNID,
            "scorpion": CreatureArchetype.ARACHNID,
            "tarantula": CreatureArchetype.ARACHNID,

            "avian": CreatureArchetype.AVIAN,
            "bird": CreatureArchetype.AVIAN,
            "harpy": CreatureArchetype.AVIAN,
            "phoenix": CreatureArchetype.AVIAN,
            "roc": CreatureArchetype.AVIAN,

            "amphibian": CreatureArchetype.AMPHIBIAN,
            "frog": CreatureArchetype.AMPHIBIAN,
            "toad": CreatureArchetype.AMPHIBIAN,
            "salamander": CreatureArchetype.AMPHIBIAN,
            "newt": CreatureArchetype.AMPHIBIAN,

            "crustacean": CreatureArchetype.CRUSTACEAN,
            "crab": CreatureArchetype.CRUSTACEAN,
            "lobster": CreatureArchetype.CRUSTACEAN,
            "shrimp": CreatureArchetype.CRUSTACEAN,

            "worm": CreatureArchetype.WORM,
            "caterpillar": CreatureArchetype.WORM,
            "centipede": CreatureArchetype.WORM,
            "millipede": CreatureArchetype.WORM,
            "leech": CreatureArchetype.WORM,

            "dragon": CreatureArchetype.DRAGON,
            "drake": CreatureArchetype.DRAGON,
            "lindworm": CreatureArchetype.DRAGON,

            # === FANTASY/MYTHICAL ===
            "demon": CreatureArchetype.DEMON,
            "devil": CreatureArchetype.DEMON,
            "fiend": CreatureArchetype.DEMON,
            "imp": CreatureArchetype.DEMON,
            "succubus": CreatureArchetype.DEMON,

            "angel": CreatureArchetype.ANGEL,
            "seraph": CreatureArchetype.ANGEL,
            "cherub": CreatureArchetype.ANGEL,
            "celestial": CreatureArchetype.ANGEL,

            "undead": CreatureArchetype.UNDEAD,
            "zombie": CreatureArchetype.UNDEAD,
            "lich": CreatureArchetype.UNDEAD,
            "ghoul": CreatureArchetype.UNDEAD,
            "revenant": CreatureArchetype.UNDEAD,
            "draugr": CreatureArchetype.UNDEAD,

            "spectral": CreatureArchetype.SPECTRAL,
            "ghost": CreatureArchetype.SPECTRAL,
            "wraith": CreatureArchetype.SPECTRAL,
            "specter": CreatureArchetype.SPECTRAL,
            "phantom": CreatureArchetype.SPECTRAL,
            "banshee": CreatureArchetype.SPECTRAL,

            "elemental": CreatureArchetype.ELEMENTAL,
            "fire_elemental": CreatureArchetype.ELEMENTAL,
            "water_elemental": CreatureArchetype.ELEMENTAL,
            "earth_elemental": CreatureArchetype.ELEMENTAL,
            "air_elemental": CreatureArchetype.ELEMENTAL,
            "djinn": CreatureArchetype.ELEMENTAL,
            "genie": CreatureArchetype.ELEMENTAL,

            # === ABERRANT ===
            "slime": CreatureArchetype.SLIME,
            "ooze": CreatureArchetype.SLIME,
            "blob": CreatureArchetype.SLIME,
            "jelly": CreatureArchetype.SLIME,
            "gelatinous": CreatureArchetype.SLIME,
            "pudding": CreatureArchetype.SLIME,

            "tentacle_beast": CreatureArchetype.TENTACLE_BEAST,
            "octopus": CreatureArchetype.TENTACLE_BEAST,
            "kraken": CreatureArchetype.TENTACLE_BEAST,
            "squid": CreatureArchetype.TENTACLE_BEAST,
            "cthulhu": CreatureArchetype.TENTACLE_BEAST,

            "multi_head": CreatureArchetype.MULTI_HEAD,
            "hydra": CreatureArchetype.MULTI_HEAD,
            "cerberus": CreatureArchetype.MULTI_HEAD,
            "ettin": CreatureArchetype.MULTI_HEAD,

            "cyclops": CreatureArchetype.CYCLOPS,
            "one_eye": CreatureArchetype.CYCLOPS,
            "monoptic": CreatureArchetype.CYCLOPS,

            "swarm": CreatureArchetype.SWARM,
            "bees": CreatureArchetype.SWARM,
            "rats": CreatureArchetype.SWARM,
            "bats": CreatureArchetype.SWARM,
            "locusts": CreatureArchetype.SWARM,
            "hive": CreatureArchetype.SWARM,

            # === CONSTRUCT/OTHER ===
            "mechanical": CreatureArchetype.MECHANICAL,
            "robot": CreatureArchetype.MECHANICAL,
            "golem": CreatureArchetype.MECHANICAL,
            "construct": CreatureArchetype.MECHANICAL,
            "automaton": CreatureArchetype.MECHANICAL,
            "clockwork": CreatureArchetype.MECHANICAL,

            "crystalline": CreatureArchetype.CRYSTALLINE,
            "crystal": CreatureArchetype.CRYSTALLINE,
            "gem": CreatureArchetype.CRYSTALLINE,
            "geode": CreatureArchetype.CRYSTALLINE,

            "fungal": CreatureArchetype.FUNGAL,
            "mushroom": CreatureArchetype.FUNGAL,
            "myconid": CreatureArchetype.FUNGAL,
            "shroom": CreatureArchetype.FUNGAL,
            "fungus": CreatureArchetype.FUNGAL,

            "plant": CreatureArchetype.PLANT,
            "treant": CreatureArchetype.PLANT,
            "ent": CreatureArchetype.PLANT,
            "vine": CreatureArchetype.PLANT,
            "dryad": CreatureArchetype.PLANT,
            "floral": CreatureArchetype.PLANT,

            "shadow": CreatureArchetype.SHADOW,
            "shade": CreatureArchetype.SHADOW,
            "darkness": CreatureArchetype.SHADOW,
            "nightshade": CreatureArchetype.SHADOW,

            "colossus": CreatureArchetype.COLOSSUS,
            "kaiju": CreatureArchetype.COLOSSUS,
            "behemoth": CreatureArchetype.COLOSSUS,
            "leviathan": CreatureArchetype.COLOSSUS,
            "tarrasque": CreatureArchetype.COLOSSUS,

            "parasitic": CreatureArchetype.PARASITIC,
            "parasite": CreatureArchetype.PARASITIC,
            "symbiote": CreatureArchetype.PARASITIC,
            "facehugger": CreatureArchetype.PARASITIC,

            "mimic": CreatureArchetype.MIMIC,
            "shapeshifter": CreatureArchetype.MIMIC,
            "doppelganger": CreatureArchetype.MIMIC,
            "changeling": CreatureArchetype.MIMIC,

            # === FALLBACK ===
            "custom": CreatureArchetype.CUSTOM,
        }
        return arch_map.get(archetype.lower(), CreatureArchetype.HUMANOID)
    
    def _segment_image(self, image_path: str, name: str, config: dict) -> Dict[str, dict]:
        """Segment image into body parts using precision segmentation"""
        parts_data = {}
        expected = config.get("expected_parts", [])
        
        # Try precision segmenter first (SAM2/OpenCV)
        if self.precision_segmenter is not None:
            try:
                rig_dir = self.output_dir / name
                segmented = self.precision_segmenter.segment_character(
                    image_path=image_path,
                    output_dir=str(rig_dir),
                    expected_parts=expected
                )
                
                for part_name, part in segmented.items():
                    parts_data[part_name] = {
                        "bbox": part.bbox,
                        "pivot": part.pivot,
                        "center": part.center,
                        "confidence": part.confidence,
                    }
                
                if parts_data:
                    logger.info(f"✅ Precision segmented {len(parts_data)} parts")
                    return parts_data
                    
            except Exception as e:
                logger.warning(f"Precision segmentation failed: {e}")
        
        # Fallback to VeilbreakersRigger
        if self.rigger is not None:
            try:
                self.rigger.load_image(image_path)
                
                prompt = " . ".join(expected)
                if prompt:
                    logger.info(f"Auto-detecting: {prompt}")
                
                for part in self.rigger.get_parts():
                    parts_data[part.name] = {
                        "bbox": part.bbox.to_xywh() if part.bbox else (0, 0, 100, 100),
                        "pivot": part.pivot if part.pivot else (50, 50),
                        "center": part.get_mask_center(),
                    }
                    
            except Exception as e:
                logger.warning(f"Auto-segmentation failed: {e}")
        
        # Final fallback to default parts
        if not parts_data:
            parts_data = self._create_default_parts(image_path, config, name)
        
        return parts_data
    
    def _create_default_parts(self, image_path: str, config: dict, name: str = None) -> Dict[str, dict]:
        """Create default part positions and ACTUALLY cut/save parts"""
        import numpy as np
        
        image = Image.open(image_path).convert("RGBA")
        w, h = image.size
        
        # Create parts directory
        if name:
            rig_dir = self.output_dir / name
            parts_dir = rig_dir / "parts"
            parts_dir.mkdir(parents=True, exist_ok=True)
        else:
            parts_dir = None
        
        parts = {}
        expected = config.get("expected_parts", ["body"])
        
        # Position parts based on common layouts
        part_positions = {
            "head": (w * 0.5, h * 0.15, w * 0.25, h * 0.2),
            "skull": (w * 0.5, h * 0.15, w * 0.25, h * 0.2),
            "torso": (w * 0.5, h * 0.4, w * 0.35, h * 0.3),
            "body": (w * 0.5, h * 0.45, w * 0.4, h * 0.4),
            "main_body": (w * 0.5, h * 0.5, w * 0.5, h * 0.5),
            "ribcage": (w * 0.5, h * 0.35, w * 0.3, h * 0.25),
            "pelvis": (w * 0.5, h * 0.55, w * 0.25, h * 0.15),
            "thorax": (w * 0.5, h * 0.4, w * 0.25, h * 0.2),
            "abdomen": (w * 0.5, h * 0.6, w * 0.3, h * 0.25),
            "arm_left": (w * 0.25, h * 0.4, w * 0.15, h * 0.35),
            "arm_right": (w * 0.75, h * 0.4, w * 0.15, h * 0.35),
            "leg_left": (w * 0.35, h * 0.75, w * 0.12, h * 0.35),
            "leg_right": (w * 0.65, h * 0.75, w * 0.12, h * 0.35),
            "leg_front_left": (w * 0.3, h * 0.6, w * 0.1, h * 0.3),
            "leg_front_right": (w * 0.7, h * 0.6, w * 0.1, h * 0.3),
            "leg_back_left": (w * 0.3, h * 0.7, w * 0.1, h * 0.25),
            "leg_back_right": (w * 0.7, h * 0.7, w * 0.1, h * 0.25),
            "wing_left": (w * 0.2, h * 0.3, w * 0.25, h * 0.4),
            "wing_right": (w * 0.8, h * 0.3, w * 0.25, h * 0.4),
            "neck": (w * 0.5, h * 0.25, w * 0.1, h * 0.1),
        }
        
        for part_name in expected:
            if part_name in part_positions:
                cx, cy, pw, ph = part_positions[part_name]
            else:
                cx, cy, pw, ph = w * 0.5, h * 0.5, w * 0.2, h * 0.2
            
            # Calculate bounding box
            x1 = int(max(0, cx - pw/2))
            y1 = int(max(0, cy - ph/2))
            x2 = int(min(w, cx + pw/2))
            y2 = int(min(h, cy + ph/2))
            
            # Actually cut and save the part if we have a parts directory
            if parts_dir and x2 > x1 and y2 > y1:
                part_img = image.crop((x1, y1, x2, y2))
                
                # Trim to actual content (non-transparent pixels)
                part_array = np.array(part_img)
                if part_array.shape[2] == 4:
                    alpha = part_array[:, :, 3]
                    rows = np.any(alpha > 10, axis=1)
                    cols = np.any(alpha > 10, axis=0)
                    
                    if np.any(rows) and np.any(cols):
                        row_min, row_max = np.nonzero(rows)[0][[0, -1]]
                        col_min, col_max = np.nonzero(cols)[0][[0, -1]]
                        
                        # Crop to actual content
                        part_img = part_img.crop((col_min, row_min, col_max + 1, row_max + 1))
                        
                        # Update coordinates
                        x1 += int(col_min)
                        y1 += int(row_min)
                
                # Save part image
                part_path = parts_dir / f"{part_name}.png"
                part_img.save(part_path)
            
            parts[part_name] = {
                "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                "pivot": (int((x2 - x1) / 2), int((y2 - y1) / 2)),
                "center": (int(cx), int(cy)),
            }
        
        return parts
    
    def _build_skeleton(self, rig: CreatureRig, parts_data: Dict, config: dict,
                        arm_count: int, leg_count: int, has_tail: bool,
                        has_wings: bool, has_hair: bool, has_cape: bool,
                        tentacle_count: int) -> None:
        """Build the bone hierarchy"""
        
        bone_config = config.get("bone_config", {})
        
        # Build bones for configured parts
        for part_name, cfg in bone_config.items():
            if part_name not in parts_data:
                # Create placeholder if part wasn't detected
                parts_data[part_name] = {
                    "bbox": (rig.width * 0.4, rig.height * 0.4, 100, 100),
                    "pivot": (50, 50),
                    "center": (rig.width / 2, rig.height / 2),
                }
            
            part_info = parts_data[part_name]
            cx, cy = part_info["center"]
            
            part_type = cfg.get("type", PartType.RIGID_CORE)
            parent = cfg.get("parent", "root")
            bone_count = cfg.get("bones", 1)
            
            bones, ik, physics = self.chain_generator.generate_chain(
                part_name=part_name,
                start_x=cx - rig.width / 2,  # Relative to root
                start_y=cy - rig.height * 0.1,
                length=part_info["bbox"][3] if isinstance(part_info["bbox"], tuple) else 100,
                parent_bone=parent,
                part_type=part_type,
                bone_count=bone_count
            )
            
            rig.bones.extend(bones)
            if ik:
                rig.ik_constraints.append(ik)
            rig.physics_constraints.extend(physics)
        
        # Add additional arms if multi-arm
        if arm_count > 2:
            self._add_extra_arms(rig, arm_count, parts_data)
        
        # Add additional legs if needed
        if leg_count > 2 and "leg_front_left" not in bone_config:
            self._add_extra_legs(rig, leg_count, parts_data)
        
        # Add optional parts
        if has_tail:
            self._add_tail(rig, parts_data)
        
        if has_wings:
            self._add_wings(rig, parts_data)
        
        if has_hair:
            self._add_hair(rig, parts_data)
        
        if has_cape:
            self._add_cape(rig, parts_data)
        
        if tentacle_count > 0:
            self._add_tentacles(rig, tentacle_count, parts_data)
        
        # Add body segments for serpent-type creatures
        body_segments = config.get("body_segments", 0)
        if body_segments > 0:
            self._add_body_segments(rig, body_segments, parts_data)
        
        # Use config leg_count if not specified by user
        config_leg_count = config.get("leg_count", 0)
        if config_leg_count > 2 and leg_count <= 2:
            self._add_insect_legs(rig, config_leg_count, parts_data)
        
        # Add ethereal trails for floating creatures
        ethereal_trails = config.get("ethereal_trails", 0)
        if ethereal_trails > 0:
            self._add_ethereal_trails(rig, ethereal_trails, parts_data)
    
    def _add_extra_arms(self, rig: CreatureRig, arm_count: int, parts_data: Dict) -> None:
        """Add extra arms for multi-arm creatures"""
        torso_center = parts_data.get("torso", {}).get("center", (rig.width / 2, rig.height * 0.4))
        
        for i in range(2, arm_count):
            side = "left" if i % 2 == 0 else "right"
            row = i // 2
            
            x_offset = -80 if side == "left" else 80
            y_offset = row * 40
            
            bones, ik, _ = self.chain_generator.generate_chain(
                part_name=f"arm_{i+1}_{side}",
                start_x=x_offset,
                start_y=torso_center[1] - rig.height * 0.1 + y_offset,
                length=100,
                angle=-45 if side == "left" else -135,
                parent_bone="torso",
                part_type=PartType.RIGID_LIMB,
                bone_count=3
            )
            
            rig.bones.extend(bones)
            if ik:
                rig.ik_constraints.append(ik)
    
    def _add_extra_legs(self, rig: CreatureRig, leg_count: int, parts_data: Dict) -> None:
        """Add extra legs for multi-legged creatures"""
        body_center = parts_data.get("body", parts_data.get("torso", {})).get(
            "center", (rig.width / 2, rig.height * 0.6))
        
        for i in range(2, leg_count):
            side = "left" if i % 2 == 0 else "right"
            position = i // 2
            
            x_offset = -50 - position * 20 if side == "left" else 50 + position * 20
            y_offset = position * 15
            
            bones, ik, _ = self.chain_generator.generate_chain(
                part_name=f"leg_{i+1}_{side}",
                start_x=x_offset,
                start_y=body_center[1] - rig.height * 0.1 + y_offset,
                length=80,
                angle=90 if side == "left" else 90,
                parent_bone="body" if "body" in parts_data else "torso",
                part_type=PartType.RIGID_LIMB,
                bone_count=3
            )
            
            rig.bones.extend(bones)
            if ik:
                rig.ik_constraints.append(ik)
    
    def _add_tail(self, rig: CreatureRig, _parts_data: Dict) -> None:
        """Add a physics-enabled tail"""
        body_bone = "body" if any(b.name == "body" for b in rig.bones) else "torso"
        
        bones, _, physics = self.chain_generator.generate_chain(
            part_name="tail",
            start_x=0,
            start_y=-20,
            length=120,
            angle=160,
            parent_bone=body_bone,
            part_type=PartType.SOFT_TENTACLE,
            preset="tail_thick",
            bone_count=5
        )
        
        rig.bones.extend(bones)
        rig.physics_constraints.extend(physics)
    
    def _add_wings(self, rig: CreatureRig, _parts_data: Dict) -> None:
        """Add wing bones"""
        body_bone = "body" if any(b.name == "body" for b in rig.bones) else "torso"
        
        for side in ["left", "right"]:
            x = -60 if side == "left" else 60
            angle = 45 if side == "left" else 135
            
            bones, ik, _ = self.chain_generator.generate_chain(
                part_name=f"wing_{side}",
                start_x=x,
                start_y=0,
                length=150,
                angle=angle,
                parent_bone=body_bone,
                part_type=PartType.RIGID_LIMB,
                bone_count=4
            )
            
            rig.bones.extend(bones)
            if ik:
                rig.ik_constraints.append(ik)
    
    def _add_hair(self, rig: CreatureRig, _parts_data: Dict) -> None:
        """Add physics-enabled hair"""
        head_bone = "head" if any(b.name == "head" for b in rig.bones) else "skull"
        
        # Add multiple hair strands
        for i, x_offset in enumerate([-20, 0, 20]):
            bones, _, physics = self.chain_generator.generate_chain(
                part_name=f"hair_{i+1}",
                start_x=x_offset,
                start_y=20,
                length=60,
                angle=-90 + x_offset,
                parent_bone=head_bone,
                part_type=PartType.SOFT_HAIR,
                preset="hair_long",
                bone_count=5
            )
            
            rig.bones.extend(bones)
            rig.physics_constraints.extend(physics)
    
    def _add_cape(self, rig: CreatureRig, _parts_data: Dict) -> None:
        """Add physics-enabled cape"""
        body_bone = "body" if any(b.name == "body" for b in rig.bones) else "torso"
        
        bones, _, physics = self.chain_generator.generate_chain(
            part_name="cape",
            start_x=0,
            start_y=10,
            length=100,
            angle=180,
            parent_bone=body_bone,
            part_type=PartType.SOFT_CLOTH,
            preset="cape_light",
            bone_count=6
        )
        
        rig.bones.extend(bones)
        rig.physics_constraints.extend(physics)
    
    def _add_tentacles(self, rig: CreatureRig, count: int, _parts_data: Dict) -> None:
        """Add physics-enabled tentacles"""
        body_bone = "main_body" if any(b.name == "main_body" for b in rig.bones) else "body"
        if not any(b.name == body_bone for b in rig.bones):
            body_bone = "torso" if any(b.name == "torso" for b in rig.bones) else "root"
        
        for i in range(count):
            angle = (360 / count) * i - 90
            x = 50 * np.cos(np.radians(angle))
            y = 50 * np.sin(np.radians(angle))
            
            bones, _, physics = self.chain_generator.generate_chain(
                part_name=f"tentacle_{i+1}",
                start_x=x,
                start_y=y,
                length=100,
                angle=angle,
                parent_bone=body_bone,
                part_type=PartType.SOFT_TENTACLE,
                preset="tentacle_slow",
                bone_count=6
            )
            
            rig.bones.extend(bones)
            rig.physics_constraints.extend(physics)
    
    def _add_body_segments(self, rig: CreatureRig, segment_count: int, parts_data: Dict) -> None:
        """Add body segments for serpent-like creatures"""
        head_center = parts_data.get("head", {}).get("center", (rig.width / 2, rig.height * 0.3))
        
        segment_length = rig.height * 0.1  # Each segment
        parent = "root"
        
        for i in range(segment_count):
            segment_name = f"body_{i+1}"
            y_pos = head_center[1] + (i * segment_length * 0.8)
            
            # Serpent body uses soft physics for flowing motion
            bones, _, physics = self.chain_generator.generate_chain(
                part_name=segment_name,
                start_x=0,
                start_y=y_pos - rig.height * 0.1,
                length=segment_length,
                angle=90,  # Pointing down
                parent_bone=parent,
                part_type=PartType.SOFT_TENTACLE if i > 0 else PartType.RIGID_CORE,
                bone_count=1
            )
            
            rig.bones.extend(bones)
            rig.physics_constraints.extend(physics)
            parent = segment_name
        
        # Update head parent to connect to first body segment
        for bone in rig.bones:
            if bone.name == "head":
                bone.parent = "body_1"
    
    def _add_insect_legs(self, rig: CreatureRig, leg_count: int, parts_data: Dict) -> None:
        """Add legs for insectoid creatures (6 or 8 legs)"""
        thorax_center = parts_data.get("thorax", {}).get(
            "center", (rig.width / 2, rig.height * 0.5))
        
        # Leg positions: pairs from front to back
        pairs = leg_count // 2
        leg_length = rig.height * 0.15
        
        for pair in range(pairs):
            for side_idx, side in enumerate(["left", "right"]):
                leg_name = f"leg_{pair+1}_{side}"
                
                # Spread legs along body
                y_offset = (pair - pairs/2 + 0.5) * 30
                x_offset = -100 if side == "left" else 100
                angle = -60 if side == "left" else -120
                
                bones, ik, _ = self.chain_generator.generate_chain(
                    part_name=leg_name,
                    start_x=x_offset,
                    start_y=thorax_center[1] - rig.height * 0.1 + y_offset,
                    length=leg_length,
                    angle=angle,
                    parent_bone="thorax",
                    part_type=PartType.RIGID_LIMB,
                    bone_count=3
                )
                
                rig.bones.extend(bones)
                if ik:
                    rig.ik_constraints.append(ik)
    
    def _add_ethereal_trails(self, rig: CreatureRig, trail_count: int, parts_data: Dict) -> None:
        """Add flowing ethereal trails for floating/spectral creatures"""
        body_center = parts_data.get("main_body", {}).get(
            "center", (rig.width / 2, rig.height * 0.4))
        
        trail_length = rig.height * 0.2
        
        for i in range(trail_count):
            trail_name = f"trail_{i+1}"
            
            # Spread trails around the body
            angle = (360 / trail_count) * i + 180  # Start from bottom
            import math
            x_offset = math.cos(math.radians(angle)) * 30
            y_offset = math.sin(math.radians(angle)) * 20
            
            bones, _, physics = self.chain_generator.generate_chain(
                part_name=trail_name,
                start_x=x_offset,
                start_y=body_center[1] - rig.height * 0.1 + y_offset,
                length=trail_length,
                angle=angle,
                parent_bone="main_body",
                part_type=PartType.SOFT_CLOTH,
                preset="ethereal",
                bone_count=4
            )
            
            rig.bones.extend(bones)
            rig.physics_constraints.extend(physics)
    
    def _add_custom_parts(self, rig: CreatureRig, custom_parts: Dict[str, dict]) -> None:
        """Add user-defined custom parts"""
        for name, cfg in custom_parts.items():
            part_type_str = cfg.get("type", "rigid_core")
            part_type = getattr(PartType, part_type_str.upper(), PartType.RIGID_CORE)
            
            bones, ik, physics = self.chain_generator.generate_chain(
                part_name=name,
                start_x=cfg.get("x", 0),
                start_y=cfg.get("y", 0),
                length=cfg.get("length", 50),
                angle=cfg.get("angle", 0),
                parent_bone=cfg.get("parent", "root"),
                part_type=part_type,
                preset=cfg.get("physics_preset"),
                bone_count=cfg.get("bones", 1)
            )
            
            rig.bones.extend(bones)
            if ik:
                rig.ik_constraints.append(ik)
            rig.physics_constraints.extend(physics)
    
    def _create_slots(self, rig: CreatureRig, _parts_data: Dict) -> None:
        """Create slots for all parts"""
        # Create a slot for each main bone (not chain bones)
        for bone in rig.bones:
            if bone.name == "root":
                continue
            if "_ik_target" in bone.name:
                continue
            # Skip chain bones (numbered suffixes > 1)
            if any(bone.name.endswith(f"_{i}") for i in range(2, 20)):
                continue
            
            slot = Slot(
                name=bone.name,
                bone=bone.name,
                attachment=bone.name
            )
            rig.slots.append(slot)
        
        # Create default skin with attachments
        rig.skins["default"] = {}
        for slot in rig.slots:
            rig.skins["default"][slot.name] = {
                slot.attachment: {
                    "type": "region",
                    "x": 0,
                    "y": 0,
                    "width": 100,
                    "height": 100,
                }
            }
    
    def _generate_animations(self, rig: CreatureRig, config: dict, speed_mult: float) -> None:
        """Generate all animations for this creature type"""
        gen = AnimationGenerator(rig)
        templates = AnimationTemplates
        
        # Gather bone groups for animation
        bone_groups = self._categorize_bones(rig)
        
        anim_list = config.get("animations", [])
        
        for anim_name in anim_list:
            try:
                anim = self._create_animation(gen, templates, anim_name, bone_groups, speed_mult)
                if anim:
                    rig.animations[anim.name] = anim
            except Exception as e:
                logger.warning(f"Failed to create animation '{anim_name}': {e}")
        
        # Add physics-based secondary animations
        self._add_physics_animations(rig, gen, templates, bone_groups)
        
        logger.info(f"Generated {len(rig.animations)} animations")
    
    def _categorize_bones(self, rig: CreatureRig) -> Dict[str, List[str]]:
        """Categorize bones into groups for animation targeting"""
        groups = {
            "root": ["root"],
            "body": [],
            "head": [],
            "arms_left": [],
            "arms_right": [],
            "arms_all": [],
            "legs_left": [],
            "legs_right": [],
            "legs_all": [],
            "legs_front": [],
            "legs_back": [],
            "tail": [],
            "wings": [],
            "hair": [],
            "cape": [],
            "tentacles": [],
            "all_slots": [],
        }
        
        for bone in rig.bones:
            name = bone.name.lower()
            
            if "head" in name or "skull" in name:
                groups["head"].append(bone.name)
            elif "torso" in name or "body" in name or "chest" in name or "ribcage" in name:
                groups["body"].append(bone.name)
            elif "arm" in name and "left" in name:
                groups["arms_left"].append(bone.name)
                groups["arms_all"].append(bone.name)
            elif "arm" in name and "right" in name:
                groups["arms_right"].append(bone.name)
                groups["arms_all"].append(bone.name)
            elif "arm" in name:
                groups["arms_all"].append(bone.name)
            elif "leg" in name and "left" in name:
                groups["legs_left"].append(bone.name)
                groups["legs_all"].append(bone.name)
                if "front" in name:
                    groups["legs_front"].append(bone.name)
                elif "back" in name:
                    groups["legs_back"].append(bone.name)
            elif "leg" in name and "right" in name:
                groups["legs_right"].append(bone.name)
                groups["legs_all"].append(bone.name)
                if "front" in name:
                    groups["legs_front"].append(bone.name)
                elif "back" in name:
                    groups["legs_back"].append(bone.name)
            elif "leg" in name:
                groups["legs_all"].append(bone.name)
            elif "tail" in name:
                groups["tail"].append(bone.name)
            elif "wing" in name:
                groups["wings"].append(bone.name)
            elif "hair" in name:
                groups["hair"].append(bone.name)
            elif "cape" in name or "cloak" in name:
                groups["cape"].append(bone.name)
            elif "tentacle" in name:
                groups["tentacles"].append(bone.name)
        
        # Populate all_slots from slots
        groups["all_slots"] = [s.name for s in rig.slots]
        
        return groups
    
    def _create_animation(self, gen: AnimationGenerator, templates, 
                          anim_name: str, groups: Dict, speed_mult: float) -> Optional[Animation]:
        """Create a single animation based on name"""
        
        # Map animation names to template methods
        anim_map = {
            # Idles
            "idle_breathe": lambda: templates.idle_breathe(
                gen, groups["body"], groups["head"], 2.0 / speed_mult),
            "idle_combat": lambda: templates.idle_combat(
                gen, "root", groups["arms_all"], 1.5 / speed_mult),
            "idle_menace": lambda: templates.idle_menace(
                gen, "root", groups["head"], 3.0 / speed_mult),
            "idle_float": lambda: templates.idle_float(
                gen, "root", 2.5 / speed_mult),
            "idle_twitch": lambda: templates.idle_twitch(
                gen, [b.name for b in gen.rig.bones if b.name != "root"], 2.0 / speed_mult),
            "idle_writhe": lambda: templates.idle_twitch(
                gen, groups["tentacles"], 2.5 / speed_mult),
            "idle_pulse": lambda: templates.idle_float(
                gen, "root", 2.0 / speed_mult),
            
            # Movement
            "walk": lambda: self._create_walk_animation(gen, templates, groups, speed_mult),
            "run": lambda: templates.run_bipedal(
                gen, "root", groups["legs_left"], groups["legs_right"],
                groups["arms_left"], groups["arms_right"], 0.5 / speed_mult),
            "slither": lambda: templates.slither(
                gen, groups["body"] + groups["tail"], 1.5 / speed_mult),
            "shamble": lambda: templates.shamble(
                gen, "root", groups["arms_all"] + groups["legs_all"], 1.5 / speed_mult),
            "drift": lambda: templates.drift(gen, "root", 2.0 / speed_mult),
            "scuttle": lambda: templates.walk_quadruped(
                gen, "root", groups["legs_front"][:1], groups["legs_front"][1:2],
                groups["legs_back"][:1], groups["legs_back"][1:2], 0.4 / speed_mult),
            
            # Attacks
            "attack_slash": lambda: templates.attack_slash(
                gen, "root", groups["arms_right"][:1] or groups["arms_all"][:1], 0.5 / speed_mult),
            "attack_thrust": lambda: templates.attack_thrust(
                gen, "root", groups["arms_right"][:1] or groups["arms_all"][:1], 0.4 / speed_mult),
            "attack_overhead": lambda: templates.attack_overhead(
                gen, "root", groups["arms_all"][:2], 0.7 / speed_mult),
            "attack_flurry": lambda: templates.attack_flurry(
                gen, "root", [groups["arms_left"], groups["arms_right"]] + 
                [[a] for a in groups["arms_all"][2:]], 1.2 / speed_mult),
            "attack_bite": lambda: templates.attack_bite(
                gen, "root", groups["head"],
                next((bone.name for bone in gen.rig.bones if "jaw" in bone.name.lower()), ""),
                0.5 / speed_mult),
            "attack_pounce": lambda: templates.attack_pounce(
                gen, "root", groups["legs_front"], 0.8 / speed_mult),
            "attack_tail_sweep": lambda: templates.attack_tail_sweep(
                gen, "root", groups["tail"], 0.6 / speed_mult),
            "attack_beam": lambda: templates.attack_beam(
                gen, "root", groups["head"], 1.5 / speed_mult),
            
            # Hit reactions
            "hit_light": lambda: templates.hit_light(
                gen, "root", groups["all_slots"][:5], 0.3 / speed_mult),
            "hit_heavy": lambda: templates.hit_heavy(
                gen, "root", groups["all_slots"][:5], 0.5 / speed_mult),
            
            # Deaths
            "death_fall_forward": lambda: templates.death_fall_forward(
                gen, "root", groups["all_slots"], 1.0 / speed_mult),
            "death_fall_backward": lambda: templates.death_fall_backward(
                gen, "root", groups["all_slots"], 1.0 / speed_mult),
            "death_dissolve": lambda: templates.death_dissolve(
                gen, "root", groups["all_slots"], 1.5 / speed_mult),
            "death_collapse": lambda: templates.death_collapse(
                gen, "root", groups["arms_all"] + groups["legs_all"], 1.2 / speed_mult),
            "death_explode": lambda: templates.death_explode(
                gen, "root", groups["all_slots"], 0.5 / speed_mult),
            
            # Specials
            "special_charge": lambda: templates.special_charge(
                gen, "root", groups["all_slots"][:3], 1.5 / speed_mult),
            "special_release": lambda: templates.special_release(
                gen, "root", groups["all_slots"][:3], 0.8 / speed_mult),
            "special_roar": lambda: templates.special_roar(
                gen, "root", groups["head"],
                next((bone.name for bone in gen.rig.bones if "jaw" in bone.name.lower()), ""),
                1.2 / speed_mult),
            "special_transform": lambda: templates.special_transform(
                gen, "root", groups["all_slots"], 2.0 / speed_mult),
            
            # Utility
            "spawn": lambda: templates.spawn(gen, "root", groups["all_slots"], 1.0 / speed_mult),
            "victory": lambda: templates.victory(gen, "root", groups["arms_all"][:2], 1.5 / speed_mult),
            "taunt": lambda: templates.taunt(gen, "root", groups["head"], 1.2 / speed_mult),
        }
        
        if anim_name in anim_map:
            return anim_map[anim_name]()
        
        return None
    
    def _create_walk_animation(self, gen: AnimationGenerator, templates,
                               groups: Dict, speed_mult: float) -> Animation:
        """Create appropriate walk animation based on creature type"""
        if groups["legs_front"] and groups["legs_back"]:
            # Quadruped
            return templates.walk_quadruped(
                gen, "root",
                groups["legs_front"][:1], groups["legs_front"][1:2] if len(groups["legs_front"]) > 1 else [],
                groups["legs_back"][:1], groups["legs_back"][1:2] if len(groups["legs_back"]) > 1 else [],
                0.8 / speed_mult
            )
        elif groups["legs_left"] and groups["legs_right"]:
            # Bipedal
            return templates.walk_bipedal(
                gen, "root",
                groups["legs_left"], groups["legs_right"],
                groups["arms_left"], groups["arms_right"],
                1.0 / speed_mult
            )
        else:
            # Floating/other
            return templates.drift(gen, "root", 2.0 / speed_mult)
    
    def _add_physics_animations(self, rig: CreatureRig, gen: AnimationGenerator,
                                templates, groups: Dict) -> None:
        """Add physics-based secondary motion animations"""
        
        if groups["hair"]:
            anim = templates.physics_hair_idle(gen, groups["hair"], 2.0)
            rig.animations["physics_hair"] = anim
        
        if groups["cape"]:
            anim = templates.physics_cape_idle(gen, groups["cape"], 3.0)
            rig.animations["physics_cape"] = anim
        
        if groups["tentacles"]:
            anim = templates.physics_tentacle_idle(gen, groups["tentacles"], 2.5)
            rig.animations["physics_tentacles"] = anim
    
    def _add_events(self, rig: CreatureRig) -> None:
        """Add standard gameplay events"""
        rig.events = {
            "hit_frame": {"int": 0, "float": 0, "string": ""},
            "screen_shake": {"int": 1, "float": 0, "string": ""},
            "spawn_start": {"int": 0, "float": 0, "string": ""},
            "spawn_complete": {"int": 0, "float": 0, "string": ""},
            "death_impact": {"int": 0, "float": 0, "string": ""},
            "death_complete": {"int": 0, "float": 0, "string": ""},
            "charge_start": {"int": 0, "float": 0, "string": ""},
            "charge_ready": {"int": 0, "float": 0, "string": ""},
            "roar_sound": {"int": 0, "float": 0, "string": ""},
            "footstep": {"int": 0, "float": 0, "string": ""},
            "attack_sound": {"int": 0, "float": 0, "string": ""},
        }
    
    def _export(self, rig: CreatureRig, image_path: str) -> str:
        """Export the complete rig"""
        # Create output directory
        rig_dir = self.output_dir / rig.name
        parts_dir = rig_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy original image as atlas placeholder
        shutil.copy(image_path, rig_dir / f"{rig.name}.png")
        
        # Write Spine JSON
        json_path = rig_dir / f"{rig.name}.json"
        json_path.write_text(rig.to_spine_json())
        
        # Write atlas file (placeholder)
        atlas_content = f"""{rig.name}.png
size: {rig.width}, {rig.height}
format: RGBA8888
filter: Linear, Linear
repeat: none
"""
        for slot in rig.slots:
            atlas_content += f"""
{slot.name}
  rotate: false
  xy: 0, 0
  size: 100, 100
  orig: 100, 100
  offset: 0, 0
  index: -1
"""
        
        atlas_path = rig_dir / f"{rig.name}.atlas"
        atlas_path.write_text(atlas_content)
        
        # Write Godot import helper
        self._write_godot_helper(rig, rig_dir)
        
        return str(json_path)
    
    def _write_godot_helper(self, rig: CreatureRig, rig_dir: Path) -> None:
        """Write a GDScript helper for using this rig in Godot"""
        script = f'''# {rig.name.upper()} - Spine Rig Helper
# Generated by VEILBREAKERS SpineRigBuilder
#
# USAGE IN GODOT:
# 1. Import the Spine runtime for Godot
# 2. Create a SpineSprite node
# 3. Set skeleton_data_res to "{rig.name}.json"
# 4. Set atlas_res to "{rig.name}.atlas"
# 5. Attach this script for easy animation control

extends Node

# Animation names available in this rig:
const ANIMATIONS = {json.dumps(list(rig.animations.keys()), indent=4)}

# Events you can connect to:
const EVENTS = {json.dumps(list(rig.events.keys()), indent=4)}

# Quick reference:
# $SpineSprite.get_animation_state().set_animation("idle_breathe", true)
# $SpineSprite.get_animation_state().add_animation("attack_slash", false, 0)
'''
        
        helper_path = rig_dir / f"{rig.name}_helper.gd"
        helper_path.write_text(script)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VEILBREAKERS Spine Rig Builder - One-Click Monster Rigging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Basic humanoid
  python spine_rig_builder.py demon.png --type humanoid
  
  # Multi-armed creature
  python spine_rig_builder.py spider_demon.png --type humanoid --arms 6
  
  # Dragon with wings and tail
  python spine_rig_builder.py dragon.png --type winged --has-tail --has-wings
  
  # Eldritch horror with tentacles
  python spine_rig_builder.py eldritch.png --type eldritch --tentacles 8
  
  # Skeleton with cape
  python spine_rig_builder.py lich.png --type skeleton --has-cape --has-hair

ARCHETYPES:
  humanoid    - Bipedal creatures (2 arms, 2 legs)
  multi_arm   - Creatures with 4-10 arms
  quadruped   - Four-legged creatures
  serpent     - Snake-like creatures
  skeleton    - Undead/skeletal creatures
  floating    - Floating/spectral creatures
  giant       - Large powerful creatures
  insectoid   - Insect-like creatures
  winged      - Creatures with wings
  eldritch    - Lovecraftian horrors
        """
    )
    
    parser.add_argument("image", help="Path to monster image")
    parser.add_argument("--name", "-n", help="Rig name (default: filename)")
    parser.add_argument("--type", "-t", default="humanoid",
                        help="Creature archetype (default: humanoid)")
    parser.add_argument("--arms", type=int, default=2, help="Number of arms")
    parser.add_argument("--legs", type=int, default=2, help="Number of legs")
    parser.add_argument("--has-tail", action="store_true", help="Include tail")
    parser.add_argument("--has-wings", action="store_true", help="Include wings")
    parser.add_argument("--has-hair", action="store_true", help="Include hair (with physics)")
    parser.add_argument("--has-cape", action="store_true", help="Include cape (with physics)")
    parser.add_argument("--tentacles", type=int, default=0, help="Number of tentacles")
    parser.add_argument("--speed", type=float, default=1.0, help="Animation speed multiplier")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    builder = SpineRigBuilder(output_dir=args.output)
    
    output = builder.build(
        image_path=args.image,
        name=args.name,
        archetype=args.type,
        arm_count=args.arms,
        leg_count=args.legs,
        has_tail=args.has_tail,
        has_wings=args.has_wings,
        has_hair=args.has_hair,
        has_cape=args.has_cape,
        tentacle_count=args.tentacles,
        animation_speed=args.speed
    )
    
    print(f"\n✅ Spine rig exported to: {output}")
    print("\nNext steps:")
    print("  1. Open the .json file in Spine editor to fine-tune")
    print("  2. Or import directly into Godot with spine-godot runtime")
    print("  3. Use the _helper.gd script for easy animation control")


if __name__ == "__main__":
    main()


# =============================================================================
# EASY ANIMATION ADDITION
# =============================================================================

def add_animation_to_rig(rig_json_path: str, animation_name: str) -> bool:
    """
    Add an animation from the library to an existing rig.
    
    Usage:
        add_animation_to_rig("output/my_monster/my_monster.json", "jump")
    """
    from animation_library import AnimationLibrary
    
    lib = AnimationLibrary()
    anim_data = lib.get(animation_name)
    
    if not anim_data:
        print(f"❌ Animation '{animation_name}' not found")
        return False
    
    # Load existing rig
    with open(rig_json_path, 'r') as f:
        rig_data = json.load(f)
    
    # Convert library format to Spine format
    spine_anim = {"bones": {}}
    
    for bone_name, timelines in anim_data.get("bones", {}).items():
        spine_anim["bones"][bone_name] = {}
        
        for timeline_type, keyframes in timelines.items():
            if timeline_type == "rotate":
                spine_anim["bones"][bone_name]["rotate"] = [
                    {"time": kf[0], "angle": kf[1]} for kf in keyframes
                ]
            elif timeline_type == "translate":
                spine_anim["bones"][bone_name]["translate"] = [
                    {"time": kf[0], "x": kf[1] if len(kf) > 1 else 0, "y": kf[2] if len(kf) > 2 else 0}
                    for kf in keyframes
                ]
            elif timeline_type == "scale":
                spine_anim["bones"][bone_name]["scale"] = [
                    {"time": kf[0], "x": kf[1] if len(kf) > 1 else 1, "y": kf[2] if len(kf) > 2 else 1}
                    for kf in keyframes
                ]
    
    # Add to rig
    rig_data["animations"][animation_name] = spine_anim
    
    # Save
    with open(rig_json_path, 'w') as f:
        json.dump(rig_data, f, indent=2)
    
    print(f"✅ Added '{animation_name}' to {rig_json_path}")
    return True


def add_custom_animation_file(rig_json_path: str, anim_json_path: str) -> bool:
    """
    Add a custom animation from a JSON file to an existing rig.
    
    Usage:
        add_custom_animation_file("output/my_monster/my_monster.json", "my_attack.json")
    """
    from animation_library import AnimationLibrary
    
    lib = AnimationLibrary()
    
    # Add custom animation to library
    if not lib.add_custom(anim_json_path):
        return False
    
    # Get the animation name
    from pathlib import Path
    anim_name = Path(anim_json_path).stem
    
    # Add to rig
    return add_animation_to_rig(rig_json_path, anim_name)


def list_available_animations():
    """List all available animations in the library"""
    from animation_library import AnimationLibrary
    lib = AnimationLibrary()
    return lib.list_all()
