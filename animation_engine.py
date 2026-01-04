#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██╗   ██╗███████╗██╗██╗     ██████╗ ██████╗ ███████╗ █████╗ ██╗  ██╗      ║
║   ██║   ██║██╔════╝██║██║     ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║ ██╔╝      ║
║   ██║   ██║█████╗  ██║██║     ██████╔╝██████╔╝█████╗  ███████║█████╔╝       ║
║   ╚██╗ ██╔╝██╔══╝  ██║██║     ██╔══██╗██╔══██╗██╔══╝  ██╔══██║██╔═██╗       ║
║    ╚████╔╝ ███████╗██║███████╗██████╔╝██║  ██║███████╗██║  ██║██║  ██╗      ║
║     ╚═══╝  ╚══════╝╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝      ║
║                                                                              ║
║              ANIMATION ENGINE v3.0 - THE ULTIMATE SPINE PIPELINE             ║
║                                                                              ║
║   • Intelligent Part Classification (Rigid vs Soft vs Physics)              ║
║   • Automatic Bone Chain Generation with IK                                 ║
║   • 100+ Pre-built Animation Templates                                      ║
║   • Physics Simulation for Hair/Cape/Tentacles                              ║
║   • Full Spine JSON Export with Baked Animations                            ║
║   • One-Click Game-Ready Output                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Created for VEILBREAKERS by Claude
"""

from __future__ import annotations
import json
import math
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from pathlib import Path
import logging

logger = logging.getLogger("VeilbreakersAnimator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '\033[36m\033[1mANIM\033[0m: \033[36m%(message)s\033[0m'
    ))
    logger.addHandler(handler)

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class PartType(Enum):
    """Classification of body parts by physics behavior"""
    RIGID_CORE = auto()      # Head, torso - no physics, single bone
    RIGID_LIMB = auto()      # Arms, legs - IK chain, no physics
    SOFT_HAIR = auto()       # Hair, mane, fur - gravity + wind
    SOFT_CLOTH = auto()      # Cape, cloak, robe - gravity + drag
    SOFT_TENTACLE = auto()   # Tentacles, tails - wave + follow
    SOFT_CHAIN = auto()      # Chains, ropes - gravity + swing
    FLOATING = auto()        # Orbs, wisps - hover + bob
    SKELETAL = auto()        # Loose bones - rattle + disconnect

class CreatureArchetype(Enum):
    """Pre-defined creature templates with animation sets"""
    # === STANDARD ARCHETYPES ===
    HUMANOID = auto()        # 2 arms, 2 legs, standard bipedal
    MULTI_ARM = auto()       # 4-10 arms, humanoid base
    QUADRUPED = auto()       # 4 legs, optional tail (wolf, horse)
    SERPENT = auto()         # No legs, body chain (snake, wyrm)
    SKELETON = auto()        # Undead, loose joints
    FLOATING = auto()        # No ground contact (ghost, wisp)
    GIANT = auto()           # Huge scale, slow power
    INSECTOID = auto()       # 6 legs, segments (beetle, mantis)
    WINGED = auto()          # Has wings (bat, dragon)
    AQUATIC = auto()         # Fins, gills (fish, shark)
    ELDRITCH = auto()        # Cosmic horror, many parts

    # === HYBRID ARCHETYPES ===
    CENTAUR = auto()         # Human upper + quadruped lower
    NAGA = auto()            # Human upper + serpent lower
    MERMAID = auto()         # Human upper + fish tail
    CHIMERA = auto()         # Mixed body parts (lion+goat+snake)

    # === CREATURE-SPECIFIC ===
    ARACHNID = auto()        # 8 legs, spider body
    AVIAN = auto()           # Bird anatomy (wings as arms)
    AMPHIBIAN = auto()       # Frog, salamander, newt
    CRUSTACEAN = auto()      # Crab, lobster (shell, claws)
    WORM = auto()            # Caterpillar, centipede (many segments)
    DRAGON = auto()          # Full dragon (4 legs + wings + tail)

    # === FANTASY/MYTHICAL ===
    DEMON = auto()           # Horns, wings, tail, cloven hooves
    ANGEL = auto()           # Humanoid with wings, halo
    UNDEAD = auto()          # Zombie, lich, ghoul
    SPECTRAL = auto()        # Ghost, wraith (transparent)
    ELEMENTAL = auto()       # Fire/water/earth/air being

    # === ABERRANT ===
    SLIME = auto()           # Formless blob, no skeleton
    TENTACLE_BEAST = auto()  # Primarily tentacles (octopus, kraken)
    MULTI_HEAD = auto()      # Hydra, cerberus (multiple heads)
    CYCLOPS = auto()         # One-eyed giant
    SWARM = auto()           # Colony (bees, rats, bats)

    # === CONSTRUCT/OTHER ===
    MECHANICAL = auto()      # Robot, golem, construct
    CRYSTALLINE = auto()     # Crystal/gem creature
    FUNGAL = auto()          # Mushroom, myconid
    PLANT = auto()           # Treant, vine creature
    SHADOW = auto()          # Shadow being, living darkness
    COLOSSUS = auto()        # Massive scale (building-sized)
    PARASITIC = auto()       # Attaches to host
    MIMIC = auto()           # Shapeshifter base form

    # === FALLBACK ===
    CUSTOM = auto()          # User-defined

class AnimationType(Enum):
    """Types of animations"""
    IDLE = auto()
    MOVEMENT = auto()
    ATTACK = auto()
    HIT = auto()
    DEATH = auto()
    SPECIAL = auto()
    UTILITY = auto()

class EaseType(Enum):
    """Animation easing functions"""
    LINEAR = "linear"
    EASE_IN = "pow2in"
    EASE_OUT = "pow2out"
    EASE_IN_OUT = "pow2"
    ELASTIC = "elastic"
    BOUNCE = "bounce"
    BACK = "back"
    STEP = "stepped"

# =============================================================================
# PHYSICS PRESETS
# =============================================================================

PHYSICS_PRESETS = {
    "hair_short": {
        "bones": 3,
        "gravity": 0.3,
        "stiffness": 0.7,
        "damping": 0.4,
        "wind_influence": 0.5,
        "wave_amplitude": 2.0,
        "wave_speed": 1.5,
    },
    "hair_long": {
        "bones": 6,
        "gravity": 0.5,
        "stiffness": 0.4,
        "damping": 0.3,
        "wind_influence": 0.7,
        "wave_amplitude": 4.0,
        "wave_speed": 1.2,
    },
    "hair_wild": {
        "bones": 8,
        "gravity": 0.2,
        "stiffness": 0.3,
        "damping": 0.2,
        "wind_influence": 1.0,
        "wave_amplitude": 8.0,
        "wave_speed": 2.0,
    },
    "cape_light": {
        "bones": 5,
        "gravity": 0.6,
        "stiffness": 0.3,
        "damping": 0.5,
        "wind_influence": 0.8,
        "wave_amplitude": 5.0,
        "wave_speed": 0.8,
    },
    "cape_heavy": {
        "bones": 4,
        "gravity": 0.8,
        "stiffness": 0.5,
        "damping": 0.6,
        "wind_influence": 0.3,
        "wave_amplitude": 3.0,
        "wave_speed": 0.5,
    },
    "tentacle_slow": {
        "bones": 8,
        "gravity": 0.1,
        "stiffness": 0.2,
        "damping": 0.3,
        "wind_influence": 0.2,
        "wave_amplitude": 10.0,
        "wave_speed": 0.5,
        "wave_offset_per_bone": 0.3,
    },
    "tentacle_fast": {
        "bones": 6,
        "gravity": 0.05,
        "stiffness": 0.4,
        "damping": 0.2,
        "wind_influence": 0.1,
        "wave_amplitude": 15.0,
        "wave_speed": 2.0,
        "wave_offset_per_bone": 0.2,
    },
    "tail_thick": {
        "bones": 5,
        "gravity": 0.4,
        "stiffness": 0.6,
        "damping": 0.4,
        "wind_influence": 0.2,
        "wave_amplitude": 8.0,
        "wave_speed": 1.0,
    },
    "tail_whip": {
        "bones": 7,
        "gravity": 0.2,
        "stiffness": 0.3,
        "damping": 0.2,
        "wind_influence": 0.3,
        "wave_amplitude": 20.0,
        "wave_speed": 3.0,
    },
    "chain": {
        "bones": 10,
        "gravity": 1.0,
        "stiffness": 0.1,
        "damping": 0.7,
        "wind_influence": 0.1,
        "swing_amplitude": 5.0,
    },
    "flame": {
        "bones": 5,
        "gravity": -0.3,  # Floats up
        "stiffness": 0.1,
        "damping": 0.1,
        "wind_influence": 0.5,
        "wave_amplitude": 8.0,
        "wave_speed": 4.0,
        "flicker": True,
    },
    "ethereal": {
        "bones": 4,
        "gravity": 0.0,
        "stiffness": 0.2,
        "damping": 0.1,
        "wind_influence": 0.3,
        "wave_amplitude": 5.0,
        "wave_speed": 0.8,
        "phase_shift": True,
    },
}

# =============================================================================
# PART CLASSIFICATION RULES
# =============================================================================

PART_CLASSIFICATION = {
    # Rigid Core Parts
    "head": PartType.RIGID_CORE,
    "skull": PartType.RIGID_CORE,
    "face": PartType.RIGID_CORE,
    "torso": PartType.RIGID_CORE,
    "body": PartType.RIGID_CORE,
    "chest": PartType.RIGID_CORE,
    "pelvis": PartType.RIGID_CORE,
    "hip": PartType.RIGID_CORE,
    "abdomen": PartType.RIGID_CORE,
    "thorax": PartType.RIGID_CORE,
    
    # Rigid Limbs
    "arm": PartType.RIGID_LIMB,
    "forearm": PartType.RIGID_LIMB,
    "upper_arm": PartType.RIGID_LIMB,
    "hand": PartType.RIGID_LIMB,
    "claw": PartType.RIGID_LIMB,
    "leg": PartType.RIGID_LIMB,
    "thigh": PartType.RIGID_LIMB,
    "shin": PartType.RIGID_LIMB,
    "foot": PartType.RIGID_LIMB,
    "wing": PartType.RIGID_LIMB,
    "wing_arm": PartType.RIGID_LIMB,
    "finger": PartType.RIGID_LIMB,
    
    # Soft - Hair/Fur
    "hair": PartType.SOFT_HAIR,
    "mane": PartType.SOFT_HAIR,
    "fur": PartType.SOFT_HAIR,
    "feather": PartType.SOFT_HAIR,
    "plume": PartType.SOFT_HAIR,
    "crest": PartType.SOFT_HAIR,
    "beard": PartType.SOFT_HAIR,
    "whisker": PartType.SOFT_HAIR,
    
    # Soft - Cloth
    "cape": PartType.SOFT_CLOTH,
    "cloak": PartType.SOFT_CLOTH,
    "robe": PartType.SOFT_CLOTH,
    "banner": PartType.SOFT_CLOTH,
    "flag": PartType.SOFT_CLOTH,
    "scarf": PartType.SOFT_CLOTH,
    "ribbon": PartType.SOFT_CLOTH,
    "cloth": PartType.SOFT_CLOTH,
    "dress": PartType.SOFT_CLOTH,
    "skirt": PartType.SOFT_CLOTH,
    "sleeve": PartType.SOFT_CLOTH,
    "tabard": PartType.SOFT_CLOTH,
    "loincloth": PartType.SOFT_CLOTH,
    
    # Soft - Tentacles
    "tentacle": PartType.SOFT_TENTACLE,
    "tail": PartType.SOFT_TENTACLE,
    "tongue": PartType.SOFT_TENTACLE,
    "vine": PartType.SOFT_TENTACLE,
    "tendril": PartType.SOFT_TENTACLE,
    "appendage": PartType.SOFT_TENTACLE,
    "trunk": PartType.SOFT_TENTACLE,
    "antenna": PartType.SOFT_TENTACLE,
    
    # Soft - Chain
    "chain": PartType.SOFT_CHAIN,
    "rope": PartType.SOFT_CHAIN,
    "whip": PartType.SOFT_CHAIN,
    "tether": PartType.SOFT_CHAIN,
    "leash": PartType.SOFT_CHAIN,
    "shackle": PartType.SOFT_CHAIN,

    # Soft - Drip/Goo (uses chain physics - gravity + swing)
    "drip": PartType.SOFT_CHAIN,
    "drips": PartType.SOFT_CHAIN,
    "goo": PartType.SOFT_CHAIN,
    "slime": PartType.SOFT_CHAIN,
    "ooze": PartType.SOFT_CHAIN,
    "droplet": PartType.SOFT_CHAIN,

    # Floating
    "orb": PartType.FLOATING,
    "eye": PartType.FLOATING,
    "wisp": PartType.FLOATING,
    "flame": PartType.FLOATING,
    "soul": PartType.FLOATING,
    "spirit": PartType.FLOATING,
    "aura": PartType.FLOATING,
    "halo": PartType.FLOATING,
    
    # Skeletal
    "bone": PartType.SKELETAL,
    "rib": PartType.SKELETAL,
    "spine_bone": PartType.SKELETAL,
    "vertebra": PartType.SKELETAL,
    "jaw": PartType.SKELETAL,
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Bone:
    """Represents a single bone in the skeleton"""
    name: str
    parent: str = ""
    x: float = 0.0
    y: float = 0.0
    length: float = 50.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    shear_x: float = 0.0
    shear_y: float = 0.0
    transform_mode: str = "normal"
    color: str = "989898FF"  # Gray in Spine
    
    # Physics properties (for soft parts)
    physics_type: Optional[str] = None
    physics_preset: Optional[str] = None
    is_ik_target: bool = False
    ik_chain_length: int = 0
    
    def to_spine_dict(self) -> dict:
        """Convert to Spine JSON format"""
        data = {"name": self.name}
        if self.parent:
            data["parent"] = self.parent
        if self.length != 0:
            data["length"] = round(self.length, 2)
        if self.x != 0:
            data["x"] = round(self.x, 2)
        if self.y != 0:
            data["y"] = round(self.y, 2)
        if self.rotation != 0:
            data["rotation"] = round(self.rotation, 2)
        if self.scale_x != 1:
            data["scaleX"] = round(self.scale_x, 3)
        if self.scale_y != 1:
            data["scaleY"] = round(self.scale_y, 3)
        if self.color != "989898FF":
            data["color"] = self.color
        return data


@dataclass
class Slot:
    """Represents a slot (draw order container) in Spine"""
    name: str
    bone: str
    attachment: str = ""
    color: str = "FFFFFFFF"
    blend: str = "normal"
    
    def to_spine_dict(self) -> dict:
        data = {"name": self.name, "bone": self.bone}
        if self.attachment:
            data["attachment"] = self.attachment
        if self.color != "FFFFFFFF":
            data["color"] = self.color
        if self.blend != "normal":
            data["blend"] = self.blend
        return data


@dataclass
class IKConstraint:
    """IK constraint for limbs"""
    name: str
    bones: List[str]
    target: str
    order: int = 0
    mix: float = 1.0
    bend_positive: bool = True
    compress: bool = False
    stretch: bool = False
    
    def to_spine_dict(self) -> dict:
        return {
            "name": self.name,
            "order": self.order,
            "bones": self.bones,
            "target": self.target,
            "mix": self.mix,
            "bendPositive": self.bend_positive,
            "compress": self.compress,
            "stretch": self.stretch,
        }


@dataclass
class PhysicsConstraint:
    """Physics constraint for soft body simulation"""
    name: str
    bone: str
    order: int = 0
    
    # Physics parameters
    inertia: float = 0.0
    strength: float = 100.0
    damping: float = 1.0
    mass_inverse: float = 1.0
    wind: float = 0.0
    gravity: float = 0.0
    mix: float = 1.0
    
    # Limits
    rotate_limit: float = 45.0
    translate_limit: float = 0.0
    scale_limit: float = 0.0
    
    def to_spine_dict(self) -> dict:
        return {
            "name": self.name,
            "order": self.order,
            "bone": self.bone,
            "inertia": self.inertia,
            "strength": self.strength,
            "damping": self.damping,
            "massInverse": self.mass_inverse,
            "wind": self.wind,
            "gravity": self.gravity,
            "mix": self.mix,
        }


@dataclass
class Keyframe:
    """Single keyframe in an animation"""
    time: float
    value: Any
    curve: str = "linear"  # linear, stepped, or bezier control points
    
    def to_spine_dict(self, value_key: str = "value") -> dict:
        data = {"time": round(self.time, 3)}
        if isinstance(self.value, (int, float)):
            data[value_key] = round(self.value, 3) if isinstance(self.value, float) else self.value
        elif isinstance(self.value, dict):
            data.update(self.value)
        else:
            data[value_key] = self.value
        if self.curve != "linear":
            data["curve"] = self.curve
        return data


@dataclass  
class BoneTimeline:
    """Animation timeline for a single bone"""
    bone_name: str
    rotate: List[Keyframe] = field(default_factory=list)
    translate: List[Keyframe] = field(default_factory=list)
    scale: List[Keyframe] = field(default_factory=list)
    shear: List[Keyframe] = field(default_factory=list)
    
    def to_spine_dict(self) -> dict:
        data = {}
        if self.rotate:
            data["rotate"] = [kf.to_spine_dict("value") for kf in self.rotate]
        if self.translate:
            data["translate"] = [kf.to_spine_dict() for kf in self.translate]
        if self.scale:
            data["scale"] = [kf.to_spine_dict() for kf in self.scale]
        if self.shear:
            data["shear"] = [kf.to_spine_dict() for kf in self.shear]
        return data


@dataclass
class SlotTimeline:
    """Animation timeline for a slot (color, attachment)"""
    slot_name: str
    color: List[Keyframe] = field(default_factory=list)
    attachment: List[Keyframe] = field(default_factory=list)
    
    def to_spine_dict(self) -> dict:
        data = {}
        if self.color:
            data["rgba"] = [kf.to_spine_dict("color") for kf in self.color]
        if self.attachment:
            data["attachment"] = [kf.to_spine_dict("name") for kf in self.attachment]
        return data


@dataclass
class Animation:
    """Complete animation with all timelines"""
    name: str
    duration: float = 1.0
    bones: Dict[str, BoneTimeline] = field(default_factory=dict)
    slots: Dict[str, SlotTimeline] = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)
    draw_order: List[dict] = field(default_factory=list)
    
    def to_spine_dict(self) -> dict:
        data = {}
        if self.bones:
            data["bones"] = {name: tl.to_spine_dict() for name, tl in self.bones.items()}
        if self.slots:
            data["slots"] = {name: tl.to_spine_dict() for name, tl in self.slots.items()}
        if self.events:
            data["events"] = self.events
        if self.draw_order:
            data["drawOrder"] = self.draw_order
        return data


@dataclass
class CreatureRig:
    """Complete creature rig with skeleton and animations"""
    name: str
    archetype: CreatureArchetype
    
    # Image dimensions
    width: int = 512
    height: int = 512
    
    # Skeleton
    bones: List[Bone] = field(default_factory=list)
    slots: List[Slot] = field(default_factory=list)
    
    # Constraints
    ik_constraints: List[IKConstraint] = field(default_factory=list)
    physics_constraints: List[PhysicsConstraint] = field(default_factory=list)
    
    # Skins and attachments
    skins: Dict[str, dict] = field(default_factory=dict)
    
    # Animations
    animations: Dict[str, Animation] = field(default_factory=dict)
    
    # Events (for gameplay hooks)
    events: Dict[str, dict] = field(default_factory=dict)
    
    def to_spine_json(self) -> str:
        """Export complete rig as Spine JSON"""
        data = {
            "skeleton": {
                "hash": self.name,
                "spine": "4.1.0",
                "x": 0,
                "y": 0, 
                "width": self.width,
                "height": self.height,
                "images": "./parts/",
                "audio": "",
            },
            "bones": [b.to_spine_dict() for b in self.bones],
            "slots": [s.to_spine_dict() for s in self.slots],
        }
        
        if self.ik_constraints:
            data["ik"] = [ik.to_spine_dict() for ik in self.ik_constraints]
        
        if self.physics_constraints:
            data["physics"] = [pc.to_spine_dict() for pc in self.physics_constraints]
        
        if self.skins:
            data["skins"] = [
                {"name": name, "attachments": attachments}
                for name, attachments in self.skins.items()
            ]
        
        if self.events:
            data["events"] = self.events
        
        if self.animations:
            data["animations"] = {
                name: anim.to_spine_dict() 
                for name, anim in self.animations.items()
            }
        
        return json.dumps(data, indent=2)


# =============================================================================
# PART CLASSIFIER
# =============================================================================

class PartClassifier:
    """Intelligently classifies body parts for appropriate physics/animation"""
    
    def __init__(self):
        self.classification_rules = PART_CLASSIFICATION.copy()
    
    def classify(self, part_name: str) -> Tuple[PartType, Optional[str]]:
        """
        Classify a part by name and return (PartType, physics_preset)
        
        Returns:
            Tuple of (PartType, physics_preset_name or None)
        """
        name_lower = part_name.lower().replace("-", "_").replace(" ", "_")
        
        # Direct match
        for keyword, part_type in self.classification_rules.items():
            if keyword in name_lower:
                preset = self._get_physics_preset(part_type, name_lower)
                return part_type, preset
        
        # Default to rigid core
        return PartType.RIGID_CORE, None
    
    def _get_physics_preset(self, part_type: PartType, name: str) -> Optional[str]:
        """Get the appropriate physics preset for a part type"""
        if part_type == PartType.SOFT_HAIR:
            if "long" in name or "mane" in name:
                return "hair_long"
            elif "wild" in name or "flame" in name:
                return "hair_wild"
            return "hair_short"
        
        elif part_type == PartType.SOFT_CLOTH:
            if "heavy" in name or "armor" in name:
                return "cape_heavy"
            return "cape_light"
        
        elif part_type == PartType.SOFT_TENTACLE:
            if "tail" in name:
                if "whip" in name or "thin" in name:
                    return "tail_whip"
                return "tail_thick"
            if "fast" in name or "attack" in name:
                return "tentacle_fast"
            return "tentacle_slow"
        
        elif part_type == PartType.SOFT_CHAIN:
            return "chain"
        
        elif part_type == PartType.FLOATING:
            if "flame" in name or "fire" in name:
                return "flame"
            if "ghost" in name or "spirit" in name:
                return "ethereal"
            return None
        
        return None
    
    def get_bone_count(self, part_type: PartType, preset: Optional[str]) -> int:
        """Get recommended bone count for a part"""
        if preset and preset in PHYSICS_PRESETS:
            return PHYSICS_PRESETS[preset].get("bones", 1)
        
        # Defaults by type
        defaults = {
            PartType.RIGID_CORE: 1,
            PartType.RIGID_LIMB: 3,
            PartType.SOFT_HAIR: 4,
            PartType.SOFT_CLOTH: 5,
            PartType.SOFT_TENTACLE: 6,
            PartType.SOFT_CHAIN: 8,
            PartType.FLOATING: 1,
            PartType.SKELETAL: 1,
        }
        return defaults.get(part_type, 1)


# =============================================================================
# BONE CHAIN GENERATOR
# =============================================================================

class BoneChainGenerator:
    """Generates bone hierarchies for different part types"""
    
    def __init__(self, classifier: PartClassifier):
        self.classifier = classifier
    
    def generate_chain(self,
                       part_name: str,
                       start_x: float,
                       start_y: float,
                       length: float,
                       angle: float = 0.0,
                       parent_bone: str = "root",
                       part_type: Optional[PartType] = None,
                       preset: Optional[str] = None,
                       bone_count: Optional[int] = None) -> Tuple[List[Bone], Optional[IKConstraint], List[PhysicsConstraint]]:
        """
        Generate a bone chain for a body part
        
        Returns:
            (bones, ik_constraint or None, physics_constraints)
        """
        if part_type is None:
            part_type, preset = self.classifier.classify(part_name)
        
        if bone_count is None:
            bone_count = self.classifier.get_bone_count(part_type, preset)
        
        bones = []
        ik = None
        physics = []
        
        if part_type == PartType.RIGID_CORE:
            bones = self._generate_single_bone(part_name, start_x, start_y, length, angle, parent_bone)
        
        elif part_type == PartType.RIGID_LIMB:
            bones, ik = self._generate_ik_chain(part_name, start_x, start_y, length, angle, parent_bone, bone_count)
        
        elif part_type in (PartType.SOFT_HAIR, PartType.SOFT_CLOTH, PartType.SOFT_TENTACLE, PartType.SOFT_CHAIN):
            bones, physics = self._generate_physics_chain(
                part_name, start_x, start_y, length, angle, parent_bone, bone_count, preset
            )
        
        elif part_type == PartType.FLOATING:
            bones = self._generate_floating_bone(part_name, start_x, start_y, parent_bone)
        
        elif part_type == PartType.SKELETAL:
            bones = self._generate_skeletal_bone(part_name, start_x, start_y, length, angle, parent_bone)
        
        return bones, ik, physics
    
    def _generate_single_bone(self, name: str, x: float, y: float, length: float, 
                               angle: float, parent: str) -> List[Bone]:
        """Single bone for rigid core parts"""
        return [Bone(
            name=name,
            parent=parent,
            x=x,
            y=y,
            length=length,
            rotation=angle,
            color="00FF00FF",  # Green for core
        )]
    
    def _generate_ik_chain(self, name: str, x: float, y: float, length: float,
                           angle: float, parent: str, count: int) -> Tuple[List[Bone], IKConstraint]:
        """Generate an IK chain for limbs"""
        bones = []
        bone_length = length / count
        
        for i in range(count):
            bone_name = f"{name}_{i+1}" if i > 0 else name
            bone_parent = bones[-1].name if bones else parent
            
            bone = Bone(
                name=bone_name,
                parent=bone_parent,
                x=0 if i > 0 else x,
                y=0 if i > 0 else y,
                length=bone_length,
                rotation=angle if i == 0 else 0,
                color="FF6600FF",  # Orange for limbs
            )
            bones.append(bone)
        
        # Add IK target bone
        target = Bone(
            name=f"{name}_ik_target",
            parent=parent,
            x=x + math.cos(math.radians(angle)) * length,
            y=y + math.sin(math.radians(angle)) * length,
            length=0,
            is_ik_target=True,
            color="FF0000FF",  # Red for IK targets
        )
        bones.append(target)
        
        # Create IK constraint
        ik = IKConstraint(
            name=f"{name}_ik",
            bones=[b.name for b in bones[:-1]],  # All bones except target
            target=target.name,
            bend_positive=True,
        )
        
        return bones, ik
    
    def _generate_physics_chain(self, name: str, x: float, y: float, length: float,
                                 angle: float, parent: str, count: int,
                                 preset: Optional[str]) -> Tuple[List[Bone], List[PhysicsConstraint]]:
        """Generate a physics-enabled bone chain"""
        bones = []
        physics = []
        bone_length = length / count
        
        # Get physics parameters
        params = PHYSICS_PRESETS.get(preset, {}) if preset else {}
        gravity = params.get("gravity", 0.5)
        stiffness = params.get("stiffness", 0.5)
        damping = params.get("damping", 0.3)
        
        for i in range(count):
            bone_name = f"{name}_{i+1}" if i > 0 else name
            bone_parent = bones[-1].name if bones else parent
            
            bone = Bone(
                name=bone_name,
                parent=bone_parent,
                x=0 if i > 0 else x,
                y=0 if i > 0 else y,
                length=bone_length,
                rotation=angle if i == 0 else 0,
                physics_type="soft",
                physics_preset=preset,
                color="00FFFFFF",  # Cyan for physics
            )
            bones.append(bone)
            
            # Add physics constraint for each bone after the first
            if i > 0:
                # Decrease stiffness along the chain for natural movement
                chain_factor = 1.0 - (i / count) * 0.5
                
                pc = PhysicsConstraint(
                    name=f"{bone_name}_physics",
                    bone=bone_name,
                    order=i,
                    inertia=1.0 - stiffness * chain_factor,
                    strength=stiffness * 100 * chain_factor,
                    damping=damping,
                    gravity=gravity * (1.0 + i * 0.2),  # More gravity further down
                    wind=params.get("wind_influence", 0.3),
                    rotate_limit=45.0 / chain_factor,
                )
                physics.append(pc)
        
        return bones, physics
    
    def _generate_floating_bone(self, name: str, x: float, y: float, parent: str) -> List[Bone]:
        """Single bone for floating parts with hover behavior"""
        return [Bone(
            name=name,
            parent=parent,
            x=x,
            y=y,
            length=0,
            color="FF00FFFF",  # Magenta for floating
        )]
    
    def _generate_skeletal_bone(self, name: str, x: float, y: float, length: float,
                                 angle: float, parent: str) -> List[Bone]:
        """Bone for skeletal/undead parts with rattle behavior"""
        return [Bone(
            name=name,
            parent=parent,
            x=x,
            y=y,
            length=length,
            rotation=angle,
            color="CCCCCCFF",  # Gray for skeletal
        )]


# =============================================================================
# ANIMATION GENERATOR BASE
# =============================================================================

class AnimationGenerator:
    """Base class for generating animations"""
    
    def __init__(self, rig: CreatureRig):
        self.rig = rig
        self.bone_lookup = {b.name: b for b in rig.bones}
    
    def create_animation(self, name: str, duration: float = 1.0) -> Animation:
        """Create a new animation"""
        return Animation(name=name, duration=duration)
    
    def add_rotation_keyframes(self, anim: Animation, bone_name: str,
                                keyframes: List[Tuple[float, float, str]]) -> None:
        """Add rotation keyframes to an animation
        
        Args:
            keyframes: List of (time, angle, curve) tuples
        """
        if bone_name not in anim.bones:
            anim.bones[bone_name] = BoneTimeline(bone_name=bone_name)
        
        for time, angle, curve in keyframes:
            anim.bones[bone_name].rotate.append(Keyframe(time, angle, curve))
    
    def add_translation_keyframes(self, anim: Animation, bone_name: str,
                                   keyframes: List[Tuple[float, float, float, str]]) -> None:
        """Add translation keyframes
        
        Args:
            keyframes: List of (time, x, y, curve) tuples
        """
        if bone_name not in anim.bones:
            anim.bones[bone_name] = BoneTimeline(bone_name=bone_name)
        
        for time, x, y, curve in keyframes:
            anim.bones[bone_name].translate.append(
                Keyframe(time, {"x": x, "y": y}, curve)
            )
    
    def add_scale_keyframes(self, anim: Animation, bone_name: str,
                            keyframes: List[Tuple[float, float, float, str]]) -> None:
        """Add scale keyframes
        
        Args:
            keyframes: List of (time, scaleX, scaleY, curve) tuples
        """
        if bone_name not in anim.bones:
            anim.bones[bone_name] = BoneTimeline(bone_name=bone_name)
        
        for time, sx, sy, curve in keyframes:
            anim.bones[bone_name].scale.append(
                Keyframe(time, {"x": sx, "y": sy}, curve)
            )
    
    def add_color_keyframes(self, anim: Animation, slot_name: str,
                            keyframes: List[Tuple[float, str, str]]) -> None:
        """Add color keyframes for flash effects
        
        Args:
            keyframes: List of (time, color_hex, curve) tuples
        """
        if slot_name not in anim.slots:
            anim.slots[slot_name] = SlotTimeline(slot_name=slot_name)
        
        for time, color, curve in keyframes:
            anim.slots[slot_name].color.append(Keyframe(time, color, curve))
    
    def add_event(self, anim: Animation, time: float, event_name: str,
                  int_value: int = 0, float_value: float = 0.0, string_value: str = "") -> None:
        """Add an event keyframe"""
        event = {"time": time, "name": event_name}
        if int_value:
            event["int"] = int_value
        if float_value:
            event["float"] = float_value
        if string_value:
            event["string"] = string_value
        anim.events.append(event)
    
    def generate_sine_wave(self, duration: float, frequency: float, amplitude: float,
                           keyframe_count: int = 20, phase: float = 0.0) -> List[Tuple[float, float, str]]:
        """Generate sine wave keyframes for organic motion"""
        keyframes = []
        for i in range(keyframe_count + 1):
            t = (i / keyframe_count) * duration
            value = math.sin((t * frequency * 2 * math.pi) + phase) * amplitude
            curve = "linear"  # Consistent easing for sine wave
            keyframes.append((t, value, curve))
        return keyframes
    
    def generate_bounce(self, start_time: float, duration: float, 
                        amplitude: float, bounces: int = 3) -> List[Tuple[float, float, str]]:
        """Generate bounce keyframes"""
        keyframes = []
        current_amp = amplitude
        current_time = start_time
        bounce_duration = duration / (bounces * 2)
        
        for _ in range(bounces):
            # Down
            keyframes.append((current_time, 0, "pow2in"))
            current_time += bounce_duration
            keyframes.append((current_time, -current_amp, "pow2out"))
            # Up
            current_time += bounce_duration
            current_amp *= 0.5  # Decay
        
        keyframes.append((start_time + duration, 0, "pow2out"))
        return keyframes


logger.info("Animation Engine Core loaded ✓")
