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
║                    MASTER CLI & CLAUDE CODE INTEGRATION                      ║
║                                                                              ║
║   The Ultimate One-Command Monster Rigging System                           ║
║                                                                              ║
║   CLI Usage:                                                                 ║
║     veilbreakers rig monster.png --type demon --arms 6 --cape --hair        ║
║                                                                              ║
║   Claude Code Usage:                                                         ║
║     "Rig this demon lord with 6 arms and flowing cape for VEILBREAKERS"     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger("VEILBREAKERS")

# =============================================================================
# NATURAL LANGUAGE PARSER FOR CLAUDE CODE
# =============================================================================

@dataclass
class RigRequest:
    """Parsed rig request from natural language or CLI"""
    image_path: str
    name: Optional[str] = None
    archetype: str = "humanoid"
    arm_count: int = 2
    leg_count: int = 2
    has_tail: bool = False
    has_wings: bool = False
    has_hair: bool = False
    has_cape: bool = False
    tentacle_count: int = 0
    animation_speed: float = 1.0
    output_format: str = "spine"  # spine, godot, both
    output_dir: str = "./output"
    
    # Additional creature details
    is_giant: bool = False
    is_skeleton: bool = False
    is_floating: bool = False
    is_serpent: bool = False
    is_insect: bool = False
    is_eldritch: bool = False


class NaturalLanguageParser:
    """
    Parses natural language descriptions into RigRequest objects
    
    Designed for Claude Code integration - understands phrases like:
    - "Rig this demon with 6 arms and a cape"
    - "Create a skeleton warrior with a sword"
    - "Animate this dragon with wings and fire breath"
    """
    
    # Keywords for archetype detection
    ARCHETYPE_KEYWORDS = {
        "humanoid": ["human", "humanoid", "person", "warrior", "knight", "soldier", 
                     "demon", "devil", "golem", "orc", "elf", "dwarf", "troll",
                     "zombie", "ghoul", "vampire", "mage", "wizard", "witch"],
        "quadruped": ["wolf", "dog", "cat", "lion", "tiger", "bear", "horse",
                      "deer", "boar", "beast", "hound", "fox", "quadruped"],
        "serpent": ["snake", "serpent", "wyrm", "naga", "lamia", "worm", "eel"],
        "skeleton": ["skeleton", "skeletal", "undead", "bones", "lich", "revenant"],
        "floating": ["ghost", "specter", "spectre", "spirit", "wraith", "phantom",
                     "wisp", "beholder", "floating", "hovering", "ethereal"],
        "giant": ["giant", "titan", "colossus", "goliath", "ogre", "cyclops", "huge"],
        "insectoid": ["spider", "insect", "bug", "beetle", "mantis", "scorpion",
                      "ant", "centipede", "millipede", "arachnid"],
        "winged": ["dragon", "wyvern", "griffin", "griffon", "phoenix", "harpy",
                   "angel", "bat", "winged", "flying"],
        "eldritch": ["eldritch", "cosmic", "lovecraft", "cthulhu", "shoggoth",
                     "aberration", "horror", "abomination", "void", "tentacled"],
        "multi_arm": ["multi-arm", "multiarm", "many-armed", "six-armed", 
                      "four-armed", "eight-armed"],
    }
    
    # Keywords for features
    FEATURE_KEYWORDS = {
        "tail": ["tail", "tailed"],
        "wings": ["wing", "wings", "winged", "flying"],
        "hair": ["hair", "mane", "flowing hair", "long hair", "wild hair"],
        "cape": ["cape", "cloak", "robe", "mantle", "shroud", "flowing"],
        "tentacles": ["tentacle", "tentacles", "appendage", "appendages"],
    }
    
    # Number words
    NUMBER_WORDS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "a": 1, "an": 1, "single": 1, "pair": 2, "couple": 2,
        "few": 3, "several": 4, "many": 6, "multiple": 4,
    }
    
    def parse(self, text: str, image_path: str) -> RigRequest:
        """Parse natural language into RigRequest"""
        text_lower = text.lower()
        
        request = RigRequest(image_path=image_path)
        
        # Detect archetype
        request.archetype = self._detect_archetype(text_lower)
        
        # Detect arm count
        request.arm_count = self._detect_count(text_lower, "arm")
        
        # Detect leg count
        request.leg_count = self._detect_count(text_lower, "leg")
        
        # Detect tentacle count
        request.tentacle_count = self._detect_count(text_lower, "tentacle")
        
        # Detect features
        request.has_tail = self._has_feature(text_lower, "tail")
        request.has_wings = self._has_feature(text_lower, "wings")
        request.has_hair = self._has_feature(text_lower, "hair")
        request.has_cape = self._has_feature(text_lower, "cape")
        
        # Special archetype flags
        request.is_giant = any(kw in text_lower for kw in self.ARCHETYPE_KEYWORDS["giant"])
        request.is_skeleton = any(kw in text_lower for kw in self.ARCHETYPE_KEYWORDS["skeleton"])
        request.is_floating = any(kw in text_lower for kw in self.ARCHETYPE_KEYWORDS["floating"])
        request.is_serpent = any(kw in text_lower for kw in self.ARCHETYPE_KEYWORDS["serpent"])
        request.is_insect = any(kw in text_lower for kw in self.ARCHETYPE_KEYWORDS["insectoid"])
        request.is_eldritch = any(kw in text_lower for kw in self.ARCHETYPE_KEYWORDS["eldritch"])
        
        # Adjust archetype based on flags
        if request.is_skeleton:
            request.archetype = "skeleton"
        elif request.is_floating:
            request.archetype = "floating"
        elif request.is_serpent:
            request.archetype = "serpent"
        elif request.is_giant:
            request.archetype = "giant"
        elif request.is_insect:
            request.archetype = "insectoid"
        elif request.is_eldritch:
            request.archetype = "eldritch"
        elif request.arm_count > 2:
            request.archetype = "multi_arm"
        
        # Animation speed modifiers
        if any(word in text_lower for word in ["slow", "lumbering", "heavy"]):
            request.animation_speed = 0.7
        elif any(word in text_lower for word in ["fast", "quick", "swift", "agile"]):
            request.animation_speed = 1.3
        
        # Output format
        if "godot" in text_lower:
            request.output_format = "godot"
        elif "both" in text_lower or "all formats" in text_lower:
            request.output_format = "both"
        else:
            request.output_format = "spine"
        
        return request
    
    def _detect_archetype(self, text: str) -> str:
        """Detect creature archetype from text"""
        for archetype, keywords in self.ARCHETYPE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return archetype
        return "humanoid"
    
    def _detect_count(self, text: str, part_type: str) -> int:
        """Detect count of a part type (e.g., "6 arms", "four-armed")"""
        import re
        
        # Look for patterns like "6 arms", "six arms", "four-armed", "multiple arms"
        patterns = [
            rf"(\d+)\s*{part_type}",       # "6 arms"
            rf"(\d+)-{part_type}",         # "6-armed"
            rf"(\w+)\s*{part_type}",       # "six arms"
            rf"(\w+)-{part_type}",         # "four-armed"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                num_str = match.group(1)
                if num_str.isdigit():
                    return int(num_str)
                elif num_str in self.NUMBER_WORDS:
                    return self.NUMBER_WORDS[num_str]
        
        # Default counts
        defaults = {"arm": 2, "leg": 2, "tentacle": 0}
        return defaults.get(part_type, 0)
    
    def _has_feature(self, text: str, feature: str) -> bool:
        """Check if text mentions a feature"""
        keywords = self.FEATURE_KEYWORDS.get(feature, [feature])
        return any(kw in text for kw in keywords)


# =============================================================================
# UNIFIED RIGGER - COMBINES ALL SYSTEMS
# =============================================================================

class VeilbreakersRigger:
    """
    The Ultimate Monster Rigging System
    
    Combines:
    - AI-powered image segmentation
    - Intelligent part classification
    - Automatic bone generation with IK
    - Physics simulation for soft parts
    - 100+ pre-built animations
    - Spine JSON export
    - Godot scene export
    - Natural language understanding
    """
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parser = NaturalLanguageParser()
        
        # Import the spine builder
        try:
            from spine_rig_builder import SpineRigBuilder
            self.spine_builder = SpineRigBuilder(str(self.output_dir))
            self._has_spine = True
        except ImportError as e:
            logger.warning(f"SpineRigBuilder not available: {e}")
            self._has_spine = False
        
        # Import the original rigger for Godot export
        try:
            from veilbreakers_rigger import VeilbreakersRigger as OriginalRigger
            self.godot_rigger = OriginalRigger(str(self.output_dir))
            self._has_godot = True
        except ImportError as e:
            logger.warning(f"Original rigger not available: {e}")
            self._has_godot = False
        
        logger.info("🐺 VEILBREAKERS Rigger initialized")
    
    def rig(self, 
            image_path: str,
            description: str = "",
            **kwargs) -> Dict[str, str]:
        """
        Rig a monster from image with optional natural language description
        
        Args:
            image_path: Path to monster image
            description: Natural language description (optional)
            **kwargs: Override any RigRequest fields
            
        Returns:
            Dict with paths to exported files
        """
        # Parse description if provided
        if description:
            request = self.parser.parse(description, image_path)
        else:
            request = RigRequest(image_path=image_path)
        
        # Apply any overrides
        for key, value in kwargs.items():
            if hasattr(request, key):
                setattr(request, key, value)
        
        # Determine name
        if not request.name:
            request.name = Path(image_path).stem
        
        logger.info(f"🎯 Rigging: {request.name}")
        logger.info(f"   Type: {request.archetype}")
        logger.info(f"   Arms: {request.arm_count}, Legs: {request.leg_count}")
        if request.has_tail:
            logger.info(f"   + Tail")
        if request.has_wings:
            logger.info(f"   + Wings")
        if request.has_hair:
            logger.info(f"   + Hair (physics)")
        if request.has_cape:
            logger.info(f"   + Cape (physics)")
        if request.tentacle_count > 0:
            logger.info(f"   + {request.tentacle_count} Tentacles")
        
        outputs = {}
        
        # Generate Spine rig
        if self._has_spine and request.output_format in ("spine", "both"):
            try:
                spine_path = self.spine_builder.build(
                    image_path=request.image_path,
                    name=request.name,
                    archetype=request.archetype,
                    arm_count=request.arm_count,
                    leg_count=request.leg_count,
                    has_tail=request.has_tail,
                    has_wings=request.has_wings,
                    has_hair=request.has_hair,
                    has_cape=request.has_cape,
                    tentacle_count=request.tentacle_count,
                    animation_speed=request.animation_speed
                )
                outputs["spine"] = spine_path
                logger.info(f"✅ Spine: {spine_path}")
            except Exception as e:
                logger.error(f"Spine export failed: {e}")
        
        # Generate Godot rig
        if self._has_godot and request.output_format in ("godot", "both"):
            try:
                # Use the original rigger for Godot export
                self.godot_rigger.load_image(request.image_path)
                # Auto-detect or use preset
                godot_path = self.godot_rigger.export()
                outputs["godot"] = godot_path
                logger.info(f"✅ Godot: {godot_path}")
            except Exception as e:
                logger.error(f"Godot export failed: {e}")
        
        return outputs
    
    def quick_rig(self, image_path: str, archetype: str = "humanoid") -> str:
        """
        Quickest possible rig - just image and type
        
        Returns path to Spine JSON
        """
        return self.rig(image_path, archetype=archetype).get("spine", "")
    
    def describe_archetypes(self) -> str:
        """Get descriptions of all available archetypes"""
        from spine_rig_builder import ARCHETYPE_CONFIGS
        
        lines = ["Available Creature Archetypes:", ""]
        for arch, config in ARCHETYPE_CONFIGS.items():
            lines.append(f"  {arch.name.lower()}: {config['description']}")
            lines.append(f"    Animations: {len(config['animations'])}")
            lines.append("")
        
        return "\n".join(lines)
    
    def list_animations(self, archetype: str = "humanoid") -> List[str]:
        """List all animations available for an archetype"""
        from spine_rig_builder import ARCHETYPE_CONFIGS, CreatureArchetype
        
        arch_map = {
            "humanoid": CreatureArchetype.HUMANOID,
            "multi_arm": CreatureArchetype.MULTI_ARM,
            "quadruped": CreatureArchetype.QUADRUPED,
            "serpent": CreatureArchetype.SERPENT,
            "skeleton": CreatureArchetype.SKELETON,
            "floating": CreatureArchetype.FLOATING,
            "giant": CreatureArchetype.GIANT,
            "insectoid": CreatureArchetype.INSECTOID,
            "winged": CreatureArchetype.WINGED,
            "eldritch": CreatureArchetype.ELDRITCH,
        }
        
        arch_enum = arch_map.get(archetype.lower(), CreatureArchetype.HUMANOID)
        config = ARCHETYPE_CONFIGS.get(arch_enum, {})
        return config.get("animations", [])


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def print_banner():
    """Print the VEILBREAKERS banner"""
    banner = """
\033[35m╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██╗   ██╗███████╗██╗██╗     ██████╗ ██████╗ ███████╗ █████╗ ██╗  ██╗      ║
║   ██║   ██║██╔════╝██║██║     ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║ ██╔╝      ║
║   ██║   ██║█████╗  ██║██║     ██████╔╝██████╔╝█████╗  ███████║█████╔╝       ║
║   ╚██╗ ██╔╝██╔══╝  ██║██║     ██╔══██╗██╔══██╗██╔══╝  ██╔══██║██╔═██╗       ║
║    ╚████╔╝ ███████╗██║███████╗██████╔╝██║  ██║███████╗██║  ██║██║  ██╗      ║
║     ╚═══╝  ╚══════╝╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝      ║
║                                                                              ║
║                    🐺 MONSTER RIGGER v3.0 - SPINE EDITION 🐺                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝\033[0m
"""
    print(banner)


def main():
    import argparse
    
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="VEILBREAKERS Monster Rigger - One-Click Production Rigging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK START:
  veilbreakers monster.png                     # Basic humanoid rig
  veilbreakers monster.png --type dragon       # Dragon with wings
  veilbreakers monster.png --describe "demon lord with 6 arms and cape"

EXAMPLES:
  # Humanoid with default animations
  veilbreakers knight.png --type humanoid
  
  # Multi-armed demon
  veilbreakers demon.png --type humanoid --arms 6 --cape --hair
  
  # Quadruped beast
  veilbreakers wolf.png --type quadruped --tail
  
  # Floating ghost
  veilbreakers ghost.png --type floating
  
  # Eldritch horror
  veilbreakers cthulhu.png --type eldritch --tentacles 8
  
  # Natural language (Claude Code style)
  veilbreakers monster.png --describe "skeleton warrior with flowing cape"

OUTPUT:
  Spine JSON (.json) - Import into Spine editor or Godot spine-runtime
  Atlas placeholder (.atlas) - Replace with actual texture atlas
  Godot helper (.gd) - Script with animation constants
        """
    )
    
    parser.add_argument("image", nargs="?", help="Path to monster image")
    parser.add_argument("--type", "-t", default="humanoid",
                        help="Creature archetype (humanoid, quadruped, dragon, etc)")
    parser.add_argument("--describe", "-d", 
                        help="Natural language description")
    parser.add_argument("--name", "-n", help="Rig name (default: filename)")
    parser.add_argument("--arms", type=int, default=2, help="Number of arms")
    parser.add_argument("--legs", type=int, default=2, help="Number of legs")
    parser.add_argument("--tail", action="store_true", help="Include tail")
    parser.add_argument("--wings", action="store_true", help="Include wings")
    parser.add_argument("--hair", action="store_true", help="Include hair (physics)")
    parser.add_argument("--cape", action="store_true", help="Include cape (physics)")
    parser.add_argument("--tentacles", type=int, default=0, help="Number of tentacles")
    parser.add_argument("--speed", type=float, default=1.0, help="Animation speed")
    parser.add_argument("--format", "-f", choices=["spine", "godot", "both"],
                        default="spine", help="Output format")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--list-types", action="store_true", 
                        help="List available creature types")
    parser.add_argument("--list-anims", 
                        help="List animations for a creature type")
    
    args = parser.parse_args()
    
    rigger = VeilbreakersRigger(output_dir=args.output)
    
    # Handle info commands
    if args.list_types:
        print(rigger.describe_archetypes())
        return
    
    if args.list_anims:
        anims = rigger.list_animations(args.list_anims)
        print(f"Animations for {args.list_anims}:")
        for anim in anims:
            print(f"  • {anim}")
        return
    
    # Require image for rigging
    if not args.image:
        parser.print_help()
        return
    
    # Build the rig
    if args.describe:
        # Natural language mode
        outputs = rigger.rig(args.image, description=args.describe)
    else:
        # Explicit mode
        outputs = rigger.rig(
            args.image,
            name=args.name,
            archetype=args.type,
            arm_count=args.arms,
            leg_count=args.legs,
            has_tail=args.tail,
            has_wings=args.wings,
            has_hair=args.hair,
            has_cape=args.cape,
            tentacle_count=args.tentacles,
            animation_speed=args.speed,
            output_format=args.format
        )
    
    print("\n" + "=" * 60)
    print("🎮 RIG COMPLETE!")
    print("=" * 60)
    
    for format_name, path in outputs.items():
        print(f"  {format_name.upper()}: {path}")
    
    print("\nNext steps:")
    print("  1. Open .json in Spine editor to adjust pivots/weights")
    print("  2. Or import directly into Godot with spine-godot")
    print("  3. Test animations: idle_breathe, attack_slash, death_fall_forward")
    print()


if __name__ == "__main__":
    main()
