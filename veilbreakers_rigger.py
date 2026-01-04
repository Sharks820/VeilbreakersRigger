#!/usr/bin/env python3
"""
????????????????????????????????????????????????????????????????????????????????????????
?                                                                                      ?
?     ???   ?????????????????     ??????? ??????? ???????? ?????? ???  ???????????    ?
?     ???   ?????????????????     ??????????????????????????????????? ????????????    ?
?     ???   ?????????  ??????     ??????????????????????  ??????????????? ??????      ?
?     ???? ??????????  ??????     ??????????????????????  ??????????????? ??????      ?
?      ??????? ??????????????????????????????  ??????????????  ??????  ???????????    ?
?       ?????  ?????????????????????????? ???  ??????????????  ??????  ???????????    ?
?                                                                                      ?
?                    ????   ???? ??????? ????   ???????????????????????????????????   ?
?                    ????? ???????????????????  ????????????????????????????????????  ?
?                    ??????????????   ????????? ???????????   ???   ??????  ????????  ?
?                    ??????????????   ?????????????????????   ???   ??????  ????????  ?
?                    ??? ??? ??????????????? ??????????????   ???   ???????????  ???  ?
?                    ???     ??? ??????? ???  ?????????????   ???   ???????????  ???  ?
?                                                                                      ?
?                         ??????? ??? ???????  ??????? ???????????????                ?
?                         ??????????????????? ???????? ????????????????               ?
?                         ??????????????  ???????  ??????????  ????????               ?
?                         ??????????????   ??????   ?????????  ????????               ?
?                         ???  ???????????????????????????????????  ???               ?
?                         ???  ?????? ???????  ??????? ???????????  ???               ?
?                                                                                      ?
?                              THE ULTIMATE CUTOUT RIG SYSTEM                          ?
?                                   AAA-QUALITY OUTPUT                                 ?
?                                                                                      ?
?  ??????????????????????????????????????????????????????????????????????????????????? ?
?                                                                                      ?
?  FEATURES:                                                                           ?
?  ? AI Auto-Detection - Just say "head. body. arms. legs." and it FINDS them        ?
?  ? Manual Precision - Click to segment overlapping/occluded parts perfectly         ?
?  ? Smart Inpainting - LaMa + Stable Diffusion for flawless gap filling             ?
?  ? Layer Management - Proper z-ordering and hierarchy                               ?
?  ? Pivot Control - Set rotation points for natural animation                        ?
?  ? Live Preview - See your rig animate before export                                ?
?  ? Godot Export - Ready-to-use .tscn scenes with animation scripts                  ?
?                                                                                      ?
?  TECH STACK:                                                                         ?
?  ? Grounded SAM 2 - State-of-the-art text-prompted detection + segmentation         ?
?  ? SAM 2.1 - Click-precise manual refinement                                        ?
?  ? LaMa - Fast, high-quality inpainting                                             ?
?  ? Stable Diffusion Inpainting - AI-powered style-matching for complex holes        ?
?                                                                                      ?
????????????????????????????????????????????????????????????????????????????????????????
"""

from __future__ import annotations
import os
import sys
import json
import math
import time
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod
import threading
from contextlib import contextmanager

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Colored log output"""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{self.BOLD}{record.levelname}{self.RESET}"
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

# Setup logger
logger = logging.getLogger("VeilbreakersRigger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(levelname)s: %(message)s'))
logger.handlers = [handler]

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class SegmentationMode(Enum):
    """Segmentation operation modes"""
    AUTO_DETECT = auto()      # AI finds parts from text
    CLICK_SELECT = auto()     # Click to select
    BOX_SELECT = auto()       # Draw box to select
    PAINT_SELECT = auto()     # Paint to select
    REFINE_ADD = auto()       # Add to current selection
    REFINE_SUBTRACT = auto()  # Remove from current selection

class InpaintQuality(Enum):
    """Inpainting quality levels"""
    FAST = auto()       # OpenCV (fastest, basic)
    STANDARD = auto()   # LaMa (fast, good quality)
    HIGH = auto()       # LaMa + refinement
    ULTRA = auto()      # Stable Diffusion (slowest, best for complex)

class ExportFormat(Enum):
    """Export format options"""
    GODOT_4 = auto()        # Godot 4.x 2D scene
    GODOT_4_3D = auto()     # Godot 4.x 3D scene with billboarded sprites  
    GODOT_3 = auto()        # Godot 3.x 2D scene
    SPINE = auto()          # Spine JSON format
    UNITY = auto()          # Unity prefab (planned)
    PNG_SEQUENCE = auto()   # Just export PNGs with metadata

# Part hierarchy templates
BODY_TEMPLATES = {
    "quadruped": {
        "name": "Quadruped (Wolf, Cat, Horse, Dragon)",
        "prompt": "head . neck . body . front left leg . front right leg . back left leg . back right leg . tail",
        "hierarchy": {
            "body": ["neck", "front left leg", "front right leg", "back left leg", "back right leg", "tail"],
            "neck": ["head"]
        },
        "z_order": ["back left leg", "back right leg", "tail", "body", "neck", "front left leg", "front right leg", "head"],
        "pivots": {
            "head": "bottom_center",
            "neck": "bottom_center", 
            "body": "center",
            "front left leg": "top_center",
            "front right leg": "top_center",
            "back left leg": "top_center",
            "back right leg": "top_center",
            "tail": "left_center"
        }
    },
    "humanoid": {
        "name": "Humanoid (Golem, Demon, Knight)",
        "prompt": "head . neck . torso . left upper arm . left forearm . left hand . right upper arm . right forearm . right hand . pelvis . left thigh . left shin . left foot . right thigh . right shin . right foot",
        "hierarchy": {
            "torso": ["neck", "left upper arm", "right upper arm", "pelvis"],
            "neck": ["head"],
            "left upper arm": ["left forearm"],
            "left forearm": ["left hand"],
            "right upper arm": ["right forearm"],
            "right forearm": ["right hand"],
            "pelvis": ["left thigh", "right thigh"],
            "left thigh": ["left shin"],
            "left shin": ["left foot"],
            "right thigh": ["right shin"],
            "right shin": ["right foot"]
        },
        "z_order": ["left thigh", "left shin", "left foot", "right thigh", "right shin", "right foot",
                    "pelvis", "torso", "left upper arm", "left forearm", "left hand",
                    "right upper arm", "right forearm", "right hand", "neck", "head"],
        "pivots": {
            "head": "bottom_center",
            "neck": "bottom_center",
            "torso": "center",
            "left upper arm": "top_right",
            "left forearm": "top_center",
            "left hand": "top_center",
            "right upper arm": "top_left",
            "right forearm": "top_center",
            "right hand": "top_center",
            "pelvis": "top_center",
            "left thigh": "top_center",
            "left shin": "top_center",
            "left foot": "top_center",
            "right thigh": "top_center",
            "right shin": "top_center",
            "right foot": "top_center"
        }
    },
    "winged": {
        "name": "Winged Creature (Dragon, Demon, Bird)",
        "prompt": "head . neck . body . left wing . right wing . left arm . right arm . left leg . right leg . tail",
        "hierarchy": {
            "body": ["neck", "left wing", "right wing", "left arm", "right arm", "left leg", "right leg", "tail"],
            "neck": ["head"]
        },
        "z_order": ["left wing", "right wing", "tail", "left leg", "right leg", "body", "neck", "left arm", "right arm", "head"],
        "pivots": {
            "head": "bottom_center",
            "neck": "bottom_center",
            "body": "center",
            "left wing": "right_center",
            "right wing": "left_center",
            "left arm": "top_right",
            "right arm": "top_left",
            "left leg": "top_center",
            "right leg": "top_center",
            "tail": "top_center"
        }
    },
    "serpent": {
        "name": "Serpent (Snake, Wyrm, Eel)",
        "prompt": "head . body segment 1 . body segment 2 . body segment 3 . body segment 4 . tail",
        "hierarchy": {
            "body segment 4": ["tail"],
            "body segment 3": ["body segment 4"],
            "body segment 2": ["body segment 3"],
            "body segment 1": ["body segment 2"],
            "head": []  # Head is root for serpent
        },
        "z_order": ["tail", "body segment 4", "body segment 3", "body segment 2", "body segment 1", "head"],
        "pivots": {
            "head": "bottom_center",
            "body segment 1": "top_center",
            "body segment 2": "top_center",
            "body segment 3": "top_center",
            "body segment 4": "top_center",
            "tail": "top_center"
        }
    },
    "spider": {
        "name": "Spider/Insectoid (8 legs)",
        "prompt": "head . thorax . abdomen . leg 1 . leg 2 . leg 3 . leg 4 . leg 5 . leg 6 . leg 7 . leg 8",
        "hierarchy": {
            "thorax": ["head", "abdomen", "leg 1", "leg 2", "leg 3", "leg 4", "leg 5", "leg 6", "leg 7", "leg 8"]
        },
        "z_order": ["leg 5", "leg 6", "leg 7", "leg 8", "abdomen", "thorax", "leg 1", "leg 2", "leg 3", "leg 4", "head"],
        "pivots": {
            "head": "bottom_center",
            "thorax": "center",
            "abdomen": "top_center",
            **{f"leg {i}": "top_center" for i in range(1, 9)}
        }
    },
    "floating": {
        "name": "Floating (Ghost, Beholder, Jellyfish)",
        "prompt": "main body . eye . tendril 1 . tendril 2 . tendril 3 . tendril 4",
        "hierarchy": {
            "main body": ["eye", "tendril 1", "tendril 2", "tendril 3", "tendril 4"]
        },
        "z_order": ["tendril 1", "tendril 2", "tendril 3", "tendril 4", "main body", "eye"],
        "pivots": {
            "main body": "center",
            "eye": "center",
            **{f"tendril {i}": "top_center" for i in range(1, 5)}
        }
    },
    "custom": {
        "name": "Custom (Define your own)",
        "prompt": "",
        "hierarchy": {},
        "z_order": [],
        "pivots": {}
    }
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Point:
    """2D Point with label (positive/negative)"""
    x: int
    y: int
    label: int = 1  # 1 = foreground, 0 = background
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

@dataclass
class BoundingBox:
    """Bounding box"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.width, self.height)
    
    def expand(self, pixels: int) -> 'BoundingBox':
        return BoundingBox(
            self.x1 - pixels, self.y1 - pixels,
            self.x2 + pixels, self.y2 + pixels
        )
    
    def clamp(self, width: int, height: int) -> 'BoundingBox':
        return BoundingBox(
            max(0, self.x1), max(0, self.y1),
            min(width, self.x2), min(height, self.y2)
        )

@dataclass
class BodyPart:
    """A segmented body part"""
    name: str
    mask: np.ndarray                    # Binary mask (H, W), values 0 or 255
    image: Optional[np.ndarray] = None  # RGBA cropped image
    bbox: Optional[BoundingBox] = None  # Bounding box in original image
    pivot: Tuple[int, int] = (0, 0)     # Pivot point relative to cropped image
    pivot_world: Tuple[int, int] = (0, 0)  # Pivot in world/original coordinates
    z_index: int = 0                    # Layer order (higher = in front)
    parent: str = ""                    # Parent part name
    children: List[str] = field(default_factory=list)
    confidence: float = 1.0             # Detection confidence
    locked: bool = False                # Prevent further editing
    visible: bool = True                # Visibility in editor
    
    # Refinement data
    positive_points: List[Point] = field(default_factory=list)
    negative_points: List[Point] = field(default_factory=list)
    
    def get_mask_center(self) -> Tuple[int, int]:
        """Get center of mass of mask"""
        coords = np.where(self.mask > 0)
        if len(coords[0]) == 0:
            return (0, 0)
        return (int(np.mean(coords[1])), int(np.mean(coords[0])))
    
    def calculate_pivot(self, pivot_type: str = "center") -> Tuple[int, int]:
        """Calculate pivot point based on type"""
        if self.bbox is None:
            return self.get_mask_center()
        
        w, h = self.bbox.width, self.bbox.height
        
        pivots = {
            "center": (w // 2, h // 2),
            "top_center": (w // 2, 0),
            "bottom_center": (w // 2, h),
            "left_center": (0, h // 2),
            "right_center": (w, h // 2),
            "top_left": (0, 0),
            "top_right": (w, 0),
            "bottom_left": (0, h),
            "bottom_right": (w, h),
        }
        
        return pivots.get(pivot_type, pivots["center"])

@dataclass  
class MonsterRig:
    """Complete monster rig with all parts and metadata"""
    name: str
    original_image: np.ndarray
    parts: Dict[str, BodyPart] = field(default_factory=dict)
    hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    root_parts: List[str] = field(default_factory=list)  # Parts with no parent
    
    # Working state
    working_image: Optional[np.ndarray] = None  # Image after parts extracted
    current_mask: Optional[np.ndarray] = None   # Current selection
    
    # Metadata
    template: str = "custom"
    created_at: str = ""
    modified_at: str = ""
    
    def add_part(self, part: BodyPart) -> None:
        """Add a part to the rig"""
        self.parts[part.name] = part
        
        if part.parent and part.parent in self.parts:
            if part.name not in self.parts[part.parent].children:
                self.parts[part.parent].children.append(part.name)
        elif not part.parent:
            if part.name not in self.root_parts:
                self.root_parts.append(part.name)
    
    def get_part(self, name: str) -> Optional[BodyPart]:
        return self.parts.get(name)
    
    def remove_part(self, name: str) -> None:
        if name in self.parts:
            part = self.parts[name]
            
            # Remove from parent's children
            if part.parent and part.parent in self.parts:
                parent = self.parts[part.parent]
                if name in parent.children:
                    parent.children.remove(name)
            
            # Remove from root_parts
            if name in self.root_parts:
                self.root_parts.remove(name)
            
            # Orphan children
            for child_name in part.children:
                if child_name in self.parts:
                    self.parts[child_name].parent = ""
                    if child_name not in self.root_parts:
                        self.root_parts.append(child_name)
            
            del self.parts[name]
    
    def get_sorted_parts(self) -> List[BodyPart]:
        """Get parts sorted by z-index"""
        return sorted(self.parts.values(), key=lambda p: p.z_index)

# =============================================================================
# SEGMENTATION ENGINE INTERFACE
# =============================================================================

class SegmentationEngine(ABC):
    """Abstract base class for segmentation engines"""
    
    @abstractmethod
    def load(self) -> None:
        """Load the model"""
        pass
    
    @abstractmethod
    def set_image(self, image: np.ndarray) -> None:
        """Set the image for segmentation"""
        pass
    
    @abstractmethod
    def segment_point(self, point: Point) -> np.ndarray:
        """Segment at a point"""
        pass
    
    @abstractmethod
    def segment_points(self, positive: List[Point], negative: List[Point]) -> np.ndarray:
        """Segment with positive and negative points"""
        pass
    
    @abstractmethod
    def segment_box(self, box: BoundingBox) -> np.ndarray:
        """Segment within a bounding box"""
        pass
    
    @abstractmethod
    def auto_detect(self, text_prompt: str) -> List[Tuple[str, np.ndarray, float]]:
        """Auto-detect parts from text prompt. Returns [(name, mask, confidence), ...]"""
        pass

# =============================================================================
# SAM 2 + GROUNDING DINO ENGINE
# =============================================================================

class GroundedSAM2Engine(SegmentationEngine):
    """
    THE ULTIMATE SEGMENTATION ENGINE
    
    Combines:
    - Grounding DINO for text-based object detection
    - SAM 2.1 for precise mask generation
    """
    
    def __init__(self, 
                 sam_size: str = "large",
                 device: str = "auto",
                 grounding_dino_config: str = None,
                 grounding_dino_checkpoint: str = None,
                 sam_checkpoint: str = None,
                 check_availability: bool = True):
        
        self.sam_size = sam_size
        self.device = self._detect_device(device)
        
        # Model paths (can be overridden)
        self.gd_config = grounding_dino_config
        self.gd_checkpoint = grounding_dino_checkpoint
        self.sam_checkpoint = sam_checkpoint
        
        # Models (lazy loaded)
        self._grounding_dino = None
        self._florence2 = None  # Florence-2 for detection AND localization
        self._florence2_processor = None
        self._sam_predictor = None
        self._sam_model = None
        self._current_image = None
        self._loaded = False
        
        # Check if SAM2 is available upfront to allow proper fallback
        if check_availability:
            self._check_sam2_available()
        
        logger.info(f"GroundedSAM2Engine initialized (device: {self.device}, SAM size: {sam_size})")
    
    def _check_sam2_available(self) -> None:
        """Check if SAM2 and PyTorch are available - raise early if not"""
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch not installed. Install with:\n"
                "  pip install torch torchvision"
            )
        
        try:
            import sam2
        except ImportError:
            raise ImportError(
                "SAM 2 not installed. Install with:\n"
                "  git clone https://github.com/facebookresearch/sam2\n"
                "  cd sam2 && pip install -e ."
            )
    
    def _detect_device(self, device: str) -> str:
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    logger.info("Apple MPS available")
                    return "mps"
            except:
                pass
            return "cpu"
        return device
    
    def load(self) -> None:
        """Load all models"""
        if self._loaded:
            return

        logger.info("Loading segmentation models...")

        # Load Florence-2 FIRST (PRIMARY - does BOTH detection AND localization)
        self._load_florence2()

        # Load Grounding DINO as BACKUP
        self._load_grounding_dino()

        # Load SAM 2
        self._load_sam2()

        self._loaded = True
        logger.info("All models loaded successfully!")

    def _load_florence2(self) -> None:
        """Load Florence-2 - Microsoft's unified vision model

        Florence-2 does BOTH detection AND localization:
        - Detects objects/parts in images
        - Returns precise bounding boxes
        - No separate tagging + localization pipeline needed
        """
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            from pathlib import Path
            import torch

            # Check for fine-tuned model first!
            finetuned_path = Path(__file__).parent / "florence2_finetuned" / "final"
            if finetuned_path.exists():
                model_id = str(finetuned_path)
                logger.info(f"🎯 Using FINE-TUNED Florence-2 from {model_id}")
            else:
                # Use Florence-2 PRO (large-ft = fine-tuned, better accuracy)
                model_id = "microsoft/Florence-2-large-ft"
                logger.info(f"Loading Florence-2 PRO from {model_id}...")

            self._florence2_processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )

            # Disable SDPA to avoid _supports_sdpa attribute error
            self._florence2 = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                attn_implementation="eager"  # Use eager attention, not SDPA
            ).to(self.device)

            self._florence2.eval()
            logger.info("Florence-2 loaded successfully")

        except ImportError:
            logger.warning("Florence-2 not available - install transformers")
            self._florence2 = None
            self._florence2_processor = None
        except Exception as e:
            logger.error(f"Failed to load Florence-2: {e}")
            self._florence2 = None
            self._florence2_processor = None

    def _load_grounding_dino(self) -> None:
        """Load Grounding DINO for text-based detection"""
        try:
            # Try the official groundingdino package
            from groundingdino.util.inference import load_model
            
            # Find config and checkpoint
            import groundingdino
            gd_pkg_path = os.path.dirname(groundingdino.__file__)
            config_paths = [
                self.gd_config,
                os.path.join(gd_pkg_path, "config", "GroundingDINO_SwinT_OGC.py"),
                "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                "configs/GroundingDINO_SwinT_OGC.py",
            ]
            
            checkpoint_paths = [
                self.gd_checkpoint,
                "checkpoints/groundingdino_swint_ogc.pth",
                "weights/groundingdino_swint_ogc.pth",
                "groundingdino_swint_ogc.pth",
            ]
            
            config_path = None
            for path in config_paths:
                if path and os.path.exists(path):
                    config_path = path
                    break
            
            checkpoint_path = None
            for path in checkpoint_paths:
                if path and os.path.exists(path):
                    checkpoint_path = path
                    break
            
            if config_path is None or checkpoint_path is None:
                logger.warning("Grounding DINO files not found - auto-detection disabled")
                logger.info("Download from: https://github.com/IDEA-Research/GroundingDINO")
                self._grounding_dino = None
                return
            
            self._grounding_dino = load_model(config_path, checkpoint_path, device=self.device)
            logger.info(f"? Grounding DINO loaded from {checkpoint_path}")
            
        except ImportError:
            logger.warning("Grounding DINO not installed - auto-detection disabled")
            logger.info("Install: pip install groundingdino-py")
            self._grounding_dino = None
        except Exception as e:
            logger.error(f"Failed to load Grounding DINO: {e}")
            self._grounding_dino = None
    
    def _load_sam2(self) -> None:
        """Load SAM 2 for precise segmentation"""
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Model configurations
            configs = {
                "tiny": ("sam2_hiera_t.yaml", ["sam2.1_hiera_tiny.pt", "sam2_hiera_tiny.pt"]),
                "small": ("sam2_hiera_s.yaml", ["sam2.1_hiera_small.pt", "sam2_hiera_small.pt"]),
                "base": ("sam2_hiera_b+.yaml", ["sam2.1_hiera_base_plus.pt", "sam2_hiera_base_plus.pt"]),
                "large": ("sam2_hiera_l.yaml", ["sam2.1_hiera_large.pt", "sam2_hiera_large.pt"]),
            }
            
            config, ckpt_names = configs.get(self.sam_size, configs["large"])
            
            # Find checkpoint
            checkpoint_path = self.sam_checkpoint
            if checkpoint_path is None:
                search_paths = ["checkpoints", "weights", "."]
                for folder in search_paths:
                    for ckpt_name in ckpt_names:
                        path = os.path.join(folder, ckpt_name)
                        if os.path.exists(path):
                            checkpoint_path = path
                            break
                    if checkpoint_path:
                        break
            
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"SAM 2 checkpoint not found. Download from: https://github.com/facebookresearch/sam2")
            
            self._sam_model = build_sam2(config, checkpoint_path, device=self.device)
            self._sam_predictor = SAM2ImagePredictor(self._sam_model)
            
            logger.info(f"? SAM 2 ({self.sam_size}) loaded from {checkpoint_path}")
            
        except ImportError:
            raise ImportError(
                "SAM 2 not installed. Install with:\n"
                "  git clone https://github.com/facebookresearch/sam2\n"
                "  cd sam2 && pip install -e ."
            )
    
    def set_image(self, image: np.ndarray) -> None:
        """Set the image for segmentation"""
        self.load()
        # Convert RGBA to RGB if needed (SAM expects RGB)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        self._current_image = image
        self._sam_predictor.set_image(image)
    
    def segment_point(self, point: Point) -> np.ndarray:
        """Segment at a single point"""
        return self.segment_points([point], [])
    
    def segment_points(self, positive: List[Point], negative: List[Point]) -> np.ndarray:
        """Segment with positive and negative points"""
        self.load()
        
        if self._current_image is None:
            raise ValueError("No image set. Call set_image first.")
        
        # Prepare points
        all_points = positive + negative
        if not all_points:
            raise ValueError("At least one point required")
        
        coords = np.array([[p.x, p.y] for p in all_points])
        labels = np.array([p.label for p in all_points])
        
        # Get masks
        masks, scores, _ = self._sam_predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True
        )
        
        # Return best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        return (mask * 255).astype(np.uint8)
    
    def segment_box(self, box: BoundingBox) -> np.ndarray:
        """Segment within a bounding box"""
        self.load()
        
        if self._current_image is None:
            raise ValueError("No image set. Call set_image first.")
        
        box_array = np.array([box.to_xyxy()])
        
        masks, scores, _ = self._sam_predictor.predict(
            box=box_array,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        return (mask * 255).astype(np.uint8)

    def segment_everything(self,
                           min_mask_area: int = 1000,
                           stability_score_thresh: float = 0.85,
                           crop_n_layers: int = 1) -> List[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        """
        Segment ALL regions in the image automatically using SAM's automatic mask generator.

        Args:
            min_mask_area: Minimum area in pixels for a valid mask
            stability_score_thresh: Minimum stability score (0-1)
            crop_n_layers: Number of crop layers for finer segmentation

        Returns:
            List of (mask, score, bbox) tuples, sorted by area (largest first)
        """
        self.load()

        if self._current_image is None:
            raise ValueError("No image set. Call set_image first.")

        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            # Fallback: use grid of points
            logger.warning("SAM2AutomaticMaskGenerator not available, using grid sampling")
            return self._segment_everything_fallback(min_mask_area)

        # Create automatic mask generator
        mask_generator = SAM2AutomaticMaskGenerator(
            model=self._sam_model,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            min_mask_region_area=min_mask_area,
        )

        # Generate all masks (ensure RGB)
        logger.info("Generating automatic masks (this may take a moment)...")
        img = self._current_image
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        masks_data = mask_generator.generate(img)

        # Convert to our format: (mask, score, bbox)
        results = []
        for m in masks_data:
            mask = (m['segmentation'].astype(np.uint8) * 255)
            score = m['stability_score']
            bbox = m['bbox']  # [x, y, w, h]
            # Convert to (x1, y1, x2, y2)
            bbox_xyxy = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            results.append((mask, score, bbox_xyxy))

        # Sort by area (largest first)
        results.sort(key=lambda x: (x[0] > 0).sum(), reverse=True)

        logger.info(f"Found {len(results)} automatic segments")
        return results

    def _segment_everything_fallback(self, min_mask_area: int = 1000) -> List[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        """Fallback: sample grid of points to find all segments"""
        h, w = self._current_image.shape[:2]
        results = []
        seen_centers = set()

        # Sample grid
        step = min(h, w) // 8
        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                try:
                    mask = self.segment_point(Point(x, y, 1))
                    area = (mask > 0).sum()

                    if area < min_mask_area:
                        continue

                    # Check if we've seen this region (by center of mass)
                    coords = np.where(mask > 0)
                    if len(coords[0]) > 0:
                        cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
                        center_key = (cy // 50, cx // 50)  # Grid quantization
                        if center_key in seen_centers:
                            continue
                        seen_centers.add(center_key)

                    # Get bounding box
                    ys, xs = np.where(mask > 0)
                    bbox = (xs.min(), ys.min(), xs.max(), ys.max())

                    results.append((mask, 0.9, bbox))
                except:
                    continue

        results.sort(key=lambda x: (x[0] > 0).sum(), reverse=True)
        return results

    def auto_detect(self, text_prompt: str,
                    box_threshold: float = 0.25,
                    text_threshold: float = 0.25) -> List[Tuple[str, np.ndarray, float]]:
        """
        Auto-detect and segment parts from text prompt
        
        Args:
            text_prompt: Parts to detect, separated by " . " 
                         Example: "head . body . arms . legs"
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            
        Returns:
            List of (part_name, mask, confidence) tuples
        """
        self.load()
        
        if self._grounding_dino is None:
            logger.warning("Grounding DINO not loaded - cannot auto-detect")
            return []
        
        if self._current_image is None:
            raise ValueError("No image set. Call set_image first.")
        
        import torch
        from groundingdino.util.inference import predict
        from groundingdino.util import box_ops
        import groundingdino.datasets.transforms as T
        
        # Transform image for Grounding DINO
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Ensure RGB (Grounding DINO expects 3 channels)
        img = self._current_image
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        pil_image = Image.fromarray(img)
        image_transformed, _ = transform(pil_image, None)
        
        # Detect
        boxes, logits, phrases = predict(
            model=self._grounding_dino,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        if len(boxes) == 0:
            logger.warning("No parts detected. Try lowering thresholds or changing prompt.")
            return []
        
        # Convert boxes to pixel coordinates
        h, w = self._current_image.shape[:2]
        boxes = boxes * torch.tensor([w, h, w, h])
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes).numpy()
        
        results = []
        name_counts = {}  # Track duplicates
        
        for box, score, phrase in zip(boxes_xyxy, logits.numpy(), phrases):
            # Get mask from SAM using the detected box
            masks, scores, _ = self._sam_predictor.predict(
                box=box.reshape(1, 4),
                multimask_output=True
            )
            
            best_idx = np.argmax(scores)
            mask = (masks[best_idx] * 255).astype(np.uint8)
            
            # Clean up name
            clean_name = phrase.strip().lower().replace(" ", "_")
            
            # Handle duplicates
            if clean_name in name_counts:
                name_counts[clean_name] += 1
                clean_name = f"{clean_name}_{name_counts[clean_name]}"
            else:
                name_counts[clean_name] = 0
            
            results.append((clean_name, mask, float(score)))
            logger.info(f"  Detected: {clean_name} (confidence: {score:.2%})")
        
        logger.info(f"Auto-detected {len(results)} parts")
        return results

    def auto_detect_florence(self,
                              box_threshold: float = 0.3) -> List[Tuple[str, np.ndarray, float]]:
        """
        PRIMARY DETECTION - Uses Florence-2 for detection AND localization.

        Florence-2 is a unified vision model that:
        1. Detects all objects/regions in the image
        2. Returns precise bounding boxes (not attention hacks)
        3. Labels each region

        Args:
            box_threshold: Confidence threshold for detection

        Returns:
            List of (part_name, mask, confidence) tuples
        """
        self.load()

        if self._florence2 is None:
            logger.warning("Florence-2 not loaded, falling back to Grounding DINO")
            if self._grounding_dino is not None:
                return self.auto_detect("head . body . arm . leg . tail . wing", box_threshold=box_threshold)
            return []

        if self._current_image is None:
            raise ValueError("No image set. Call set_image first.")

        import torch
        from PIL import Image

        logger.info("Florence-2 detecting regions...")

        # Body parts we care about
        BODY_PARTS = {
            'head', 'face', 'skull', 'eye', 'eyes', 'mouth', 'nose', 'ear', 'horn', 'horns',
            'body', 'torso', 'chest', 'stomach', 'back', 'trunk',
            'arm', 'arms', 'hand', 'hands', 'finger', 'fingers', 'claw', 'claws', 'fist',
            'leg', 'legs', 'foot', 'feet', 'paw', 'paws', 'hoof',
            'tail', 'wing', 'wings', 'fin', 'tentacle', 'tentacles',
            'hair', 'fur', 'mane', 'feather', 'scale',
            'weapon', 'sword', 'axe', 'staff', 'shield', 'spear',
            'armor', 'helmet', 'cape', 'cloak',
            'shoulder', 'neck', 'jaw', 'knee', 'elbow',
            'creature', 'monster', 'beast', 'demon', 'dragon', 'character', 'figure', 'person',
        }

        # Handle both PIL and numpy input
        if isinstance(self._current_image, Image.Image):
            pil_image = self._current_image
            np_image = np.array(pil_image)
        else:
            np_image = self._current_image
            pil_image = Image.fromarray(np_image)

        # Convert RGBA to RGB if needed (models expect RGB)
        if np_image.ndim == 3 and np_image.shape[2] == 4:
            np_image = np_image[:, :, :3]
            pil_image = Image.fromarray(np_image)

        h, w = np_image.shape[:2]

        # Ensure SAM has the image set for mask prediction
        self._sam_predictor.set_image(np_image)

        # Use CAPTION_TO_PHRASE_GROUNDING - tell Florence-2 what body parts to find
        # COMPLETE body parts list - includes all variations for learning
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        body_parts_prompt = (
            # Head/face parts
            "head. face. eye. eyes. mouth. jaw. teeth. nose. horn. ear. skull. "
            # Upper body
            "neck. chest. torso. body. back. stomach. belly. "
            # Arms & hands (all variations)
            "shoulder. arm. arms. elbow. wrist. hand. hands. finger. fingers. fist. palm. claw. claws. "
            # Legs & feet (all variations)
            "hip. leg. legs. thigh. knee. ankle. foot. feet. toe. paw. hoof. "
            # Extras
            "tail. wing. wings. tentacle. "
            # Appearance
            "hair. beard. mane. fur. scales. "
            # Equipment
            "armor. helmet. weapon. sword. shield. cape."
        )

        inputs = self._florence2_processor(
            text=task + body_parts_prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)

        logger.info(f"Florence-2 searching for: {body_parts_prompt}")

        with torch.no_grad():
            generated_ids = self._florence2.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].to(self._florence2.dtype),
                max_new_tokens=1024,
                num_beams=1,
                do_sample=False,
                use_cache=False  # Disable KV cache - fixes bugs in Florence-2
            )

        generated_text = self._florence2_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Parse Florence-2 output
        parsed = self._florence2_processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(w, h)
        )

        # Extract bboxes and labels
        bboxes = parsed.get(task, {}).get('bboxes', [])
        labels = parsed.get(task, {}).get('labels', [])

        if not bboxes:
            logger.warning("Florence-2 found nothing, using SAM Everything")
            segments = self.segment_everything(min_mask_area=500)
            return [(f"part_{i}", mask, score) for i, (mask, score, _) in enumerate(segments[:8])]

        logger.info(f"Florence-2 found {len(bboxes)} regions: {labels}")

        results = []
        name_counts = {}

        for bbox, label in zip(bboxes, labels):
            label_lower = label.lower().strip()

            # CRITICAL: Extract body part name from descriptive label
            # "purple cat head" -> "head", "monster" -> "body"
            body_part = self._extract_body_part(label_lower)

            # Skip non-body-part detections
            if body_part.startswith("unknown_"):
                logger.info(f"  SKIP: '{label_lower}' -> not a body part")
                continue

            x1, y1, x2, y2 = bbox
            box_array = np.array([x1, y1, x2, y2])

            # Use SAM to get precise mask
            masks, scores, _ = self._sam_predictor.predict(
                box=box_array.reshape(1, 4),
                multimask_output=True
            )

            best_idx = np.argmax(scores)
            mask = (masks[best_idx] * 255).astype(np.uint8)
            confidence = float(scores[best_idx])

            # Handle duplicates (arm, arm_2, arm_3)
            if body_part in name_counts:
                name_counts[body_part] += 1
                final_name = f"{body_part}_{name_counts[body_part]}"
            else:
                name_counts[body_part] = 1
                final_name = body_part

            results.append((final_name, mask, confidence))
            logger.info(f"  '{label_lower}' -> {final_name}: bbox={[int(x) for x in bbox]}, conf={confidence:.2%}")

        return results

    def _extract_body_part(self, label: str) -> str:
        """
        Convert a descriptive label to a body part name.
        'purple cat head' -> 'head'
        'monster' -> 'body'
        """
        # Check if any body part is mentioned
        for part in BODY_PARTS:
            if part in label:
                return part

        # Mapping for full-object descriptions
        mappings = {
            "cat": "body", "dog": "body", "monster": "body", "creature": "body",
            "animal": "body", "character": "body", "figure": "body", "person": "body",
            "dragon": "body", "beast": "body", "demon": "body",
            "face": "head", "skull": "head", "torso": "body", "trunk": "body",
        }

        for key, value in mappings.items():
            if key in label:
                return value

        # Try last word
        words = label.split()
        if words and words[-1] in BODY_PARTS:
            return words[-1]

        return f"unknown_{label.replace(' ', '_')[:15]}"

    def smart_detect(self, text_prompt: str = None,
                     use_florence: bool = True,
                     box_threshold: float = 0.20) -> List[Tuple[str, np.ndarray, float]]:
        """
        BEST detection - uses Florence-2 unified vision model.

        Florence-2 is Microsoft's state-of-the-art vision model that does
        BOTH detection AND localization in one pass. No hacks, no attention
        extraction - it was TRAINED to output bounding boxes.

        Priority:
        1. Florence-2 dense region captioning (finds AND locates all parts)
        2. Grounding DINO with text prompt (backup)
        3. SAM Everything mode (last resort)

        Args:
            text_prompt: Optional custom prompt for Grounding DINO fallback
            use_florence: Whether to try Florence-2 first (default: True)
            box_threshold: Detection confidence threshold

        Returns:
            List of (part_name, mask, confidence) tuples
        """
        self.load()

        # PRIMARY: Florence-2 - unified detection + localization
        if use_florence and self._florence2 is not None:
            logger.info("Using Florence-2 unified detection...")
            results = self.auto_detect_florence(box_threshold=box_threshold)
            if results:
                return results
            logger.info("Florence-2 found nothing, trying fallbacks...")

        # FALLBACK 1: Grounding DINO with text prompt
        if self._grounding_dino is not None:
            prompt = text_prompt if text_prompt else "head . body . arm . leg . tail . wing . weapon . hand . foot . face"
            logger.info(f"Trying Grounding DINO with: {prompt}")
            results = self.auto_detect(prompt, box_threshold=box_threshold)
            if results:
                return results

        # FALLBACK 2: SAM Everything mode
        logger.info("Using SAM Everything mode")
        segments = self.segment_everything(min_mask_area=500)

        if not segments:
            return []

        # Name by position
        results = []
        segments_sorted = sorted(segments, key=lambda s: -np.sum(s[0] > 0))

        if len(segments_sorted) > 0:
            body_seg = segments_sorted[0]
            results.append(("body", body_seg[0], body_seg[1]))

            remaining = segments_sorted[1:8]
            if remaining:
                remaining_by_y = sorted(remaining, key=lambda s: s[2][1])
                results.append(("head", remaining_by_y[0][0], remaining_by_y[0][1]))

                for i, (mask, score, bbox) in enumerate(remaining_by_y[1:]):
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2
                    img_w = self._current_image.shape[1]

                    if cx < img_w * 0.4:
                        name = "arm_left" if y1 < self._current_image.shape[0] * 0.6 else "leg_left"
                    elif cx > img_w * 0.6:
                        name = "arm_right" if y1 < self._current_image.shape[0] * 0.6 else "leg_right"
                    else:
                        name = f"part_{i}"

                    existing_names = [r[0] for r in results]
                    if name in existing_names:
                        name = f"{name}_{i}"

                    results.append((name, mask, score))

        return results

# =============================================================================
# FALLBACK: OPENCV-BASED SEGMENTATION
# =============================================================================

class OpenCVSegmentationEngine(SegmentationEngine):
    """Fallback segmentation using OpenCV when SAM2 is not available"""
    
    def __init__(self):
        self._current_image = None
        self._hsv_image = None
        self._edges = None
    
    def load(self) -> None:
        pass  # No model to load
    
    def set_image(self, image: np.ndarray) -> None:
        import cv2
        self._current_image = image.copy()
        self._hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Pre-compute edges
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self._edges = cv2.Canny(gray, 50, 150)
    
    def segment_point(self, point: Point) -> np.ndarray:
        return self.segment_points([point], [])
    
    def segment_points(self, positive: List[Point], negative: List[Point]) -> np.ndarray:
        """Use flood fill from positive points, exclude negative"""
        import cv2
        
        if self._current_image is None:
            raise ValueError("No image set")
        
        h, w = self._current_image.shape[:2]
        
        # Start with flood fill from first positive point
        if not positive:
            return np.zeros((h, w), dtype=np.uint8)
        
        # Create mask for flood fill (needs to be 2 pixels larger)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Flood fill from each positive point
        for point in positive:
            temp_mask = np.zeros((h + 2, w + 2), np.uint8)
            flood_img = self._current_image.copy()
            
            cv2.floodFill(
                flood_img, temp_mask,
                (point.x, point.y),
                (255, 255, 255),
                loDiff=(32, 32, 32),
                upDiff=(32, 32, 32),
                flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8)
            )
            
            mask = cv2.bitwise_or(mask, temp_mask)
        
        # Remove regions around negative points
        for point in negative:
            temp_mask = np.zeros((h + 2, w + 2), np.uint8)
            flood_img = self._current_image.copy()
            
            cv2.floodFill(
                flood_img, temp_mask,
                (point.x, point.y),
                (255, 255, 255),
                loDiff=(32, 32, 32),
                upDiff=(32, 32, 32),
                flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8)
            )
            
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(temp_mask))
        
        # Trim the 1-pixel border
        result = mask[1:-1, 1:-1]
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        
        return result
    
    def segment_box(self, box: BoundingBox) -> np.ndarray:
        """Use GrabCut for box-based segmentation"""
        import cv2
        
        if self._current_image is None:
            raise ValueError("No image set")
        
        h, w = self._current_image.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        rect = (box.x1, box.y1, box.width, box.height)
        
        cv2.grabCut(
            self._current_image, mask, rect,
            bgd_model, fgd_model,
            5, cv2.GC_INIT_WITH_RECT
        )
        
        result = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255, 0
        ).astype(np.uint8)
        
        return result
    
    def auto_detect(self, text_prompt: str) -> List[Tuple[str, np.ndarray, float]]:
        logger.warning("Auto-detection not available with OpenCV engine")
        return []

# =============================================================================
# INPAINTING ENGINE
# =============================================================================

class InpaintingEngine:
    """
    Multi-backend inpainting engine
    
    Supports:
    - OpenCV (fast, basic)
    - LaMa (fast, high quality)
    - Stable Diffusion (slow, best for complex scenes)
    """
    
    def __init__(self):
        self._lama = None
        self._sd_pipeline = None
        self._loaded_backends = set()
    
    def _load_lama(self) -> bool:
        """Load LaMa inpainting model"""
        if "lama" in self._loaded_backends:
            return self._lama is not None
        
        try:
            from simple_lama_inpainting import SimpleLama
            self._lama = SimpleLama()
            self._loaded_backends.add("lama")
            logger.info("? LaMa inpainting loaded")
            return True
        except ImportError:
            logger.warning("LaMa not installed: pip install simple-lama-inpainting")
            self._loaded_backends.add("lama")
            return False
    
    def _load_sd(self) -> bool:
        """Load Stable Diffusion inpainting pipeline"""
        if "sd" in self._loaded_backends:
            return self._sd_pipeline is not None
        
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline
            
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self._sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=dtype,
                safety_checker=None
            )
            
            if torch.cuda.is_available():
                self._sd_pipeline = self._sd_pipeline.to("cuda")
            
            self._loaded_backends.add("sd")
            logger.info("? Stable Diffusion inpainting loaded")
            return True
        except Exception as e:
            logger.warning(f"SD Inpainting not available: {e}")
            self._loaded_backends.add("sd")
            return False
    
    def inpaint(self, 
                image: np.ndarray, 
                mask: np.ndarray,
                quality: InpaintQuality = InpaintQuality.STANDARD,
                prompt: str = None) -> np.ndarray:
        """
        Inpaint the masked region
        
        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W), 255 = inpaint
            quality: Inpainting quality level
            prompt: Text prompt for SD inpainting
            
        Returns:
            Inpainted RGB image
        """
        if quality == InpaintQuality.FAST:
            return self._inpaint_opencv(image, mask)
        
        if quality == InpaintQuality.STANDARD or quality == InpaintQuality.HIGH:
            if self._load_lama():
                result = self._inpaint_lama(image, mask)
                if quality == InpaintQuality.HIGH:
                    # Second pass for refinement
                    result = self._inpaint_lama(result, mask)
                return result
            else:
                return self._inpaint_opencv(image, mask)
        
        if quality == InpaintQuality.ULTRA:
            if self._load_sd() and prompt:
                return self._inpaint_sd(image, mask, prompt)
            elif self._load_lama():
                return self._inpaint_lama(image, mask)
            else:
                return self._inpaint_opencv(image, mask)
        
        return self._inpaint_opencv(image, mask)
    
    def _inpaint_opencv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fast inpainting with OpenCV"""
        import cv2
        
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Dilate mask for better coverage
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Use Telea algorithm
        result = cv2.inpaint(bgr, dilated_mask, 5, cv2.INPAINT_TELEA)
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    def _inpaint_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """High-quality inpainting with LaMa"""
        pil_image = Image.fromarray(image)
        pil_mask = Image.fromarray(mask).convert('L')
        
        result = self._lama(pil_image, pil_mask)
        return np.array(result)
    
    def _inpaint_sd(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
        """AI inpainting with Stable Diffusion"""
        # Resize for SD (works best at 512x512)
        h, w = image.shape[:2]
        
        pil_image = Image.fromarray(image).resize((512, 512))
        pil_mask = Image.fromarray(mask).convert('L').resize((512, 512))
        
        # Generate
        result = self._sd_pipeline(
            prompt=prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        # Resize back
        result = result.resize((w, h), Image.LANCZOS)
        
        return np.array(result)

# =============================================================================
# PART EXTRACTOR
# =============================================================================

class PartExtractor:
    """Extracts body parts from images with proper cropping and pivot calculation"""
    
    def __init__(self, inpainter: InpaintingEngine):
        self.inpainter = inpainter
    
    def extract(self,
                image: np.ndarray,
                mask: np.ndarray,
                name: str,
                pivot_type: str = "center",
                z_index: int = 0,
                parent: str = "",
                inpaint: bool = True,
                inpaint_quality: InpaintQuality = InpaintQuality.STANDARD,
                inpaint_prompt: str = None,
                padding: int = 5) -> Tuple[BodyPart, np.ndarray]:
        """
        Extract a body part from the image
        
        Args:
            image: RGB image
            mask: Binary mask
            name: Part name
            pivot_type: Pivot point type ("center", "top_center", etc.)
            z_index: Layer order
            parent: Parent part name
            inpaint: Whether to inpaint the hole
            inpaint_quality: Inpainting quality level
            inpaint_prompt: Prompt for SD inpainting
            padding: Pixels to add around bounding box
            
        Returns:
            (BodyPart, inpainted_image)
        """
        h, w = image.shape[:2]
        
        # Find bounding box from mask
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            raise ValueError(f"Empty mask for part: {name}")
        
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        
        y_min, y_max = y_indices[0], y_indices[-1]
        x_min, x_max = x_indices[0], x_indices[-1]
        
        # Add padding
        y_min = max(0, y_min - padding)
        y_max = min(h - 1, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(w - 1, x_max + padding)
        
        bbox = BoundingBox(x_min, y_min, x_max, y_max)
        
        # Create RGBA image
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = mask
        
        # Crop
        part_image = rgba[y_min:y_max+1, x_min:x_max+1].copy()
        
        # Calculate pivot in local coordinates
        part = BodyPart(
            name=name,
            mask=mask,
            image=part_image,
            bbox=bbox,
            z_index=z_index,
            parent=parent
        )
        
        part.pivot = part.calculate_pivot(pivot_type)
        part.pivot_world = (x_min + part.pivot[0], y_min + part.pivot[1])
        
        # Inpaint if requested
        if inpaint:
            inpainted = self.inpainter.inpaint(
                image, mask,
                quality=inpaint_quality,
                prompt=inpaint_prompt
            )
        else:
            inpainted = image.copy()
        
        return part, inpainted

# =============================================================================
# GODOT EXPORTER
# =============================================================================

class GodotExporter:
    """Exports monster rigs to Godot-ready format"""
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
    
    def export(self, 
               rig: MonsterRig,
               format: ExportFormat = ExportFormat.GODOT_4) -> str:
        """
        Export a complete monster rig
        
        Creates:
        - parts/ folder with PNG images
        - {name}.tscn scene file
        - {name}_rig.gd animator script
        - rig_data.json metadata
        
        Returns:
            Path to the main scene file
        """
        monster_dir = self.output_dir / rig.name
        parts_dir = monster_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        
        # Export part images
        for name, part in rig.parts.items():
            if part.image is not None:
                path = parts_dir / f"{name}.png"
                Image.fromarray(part.image).save(path, "PNG")
                logger.info(f"  Saved: {name}.png")
        
        # Generate scene file based on format
        if format == ExportFormat.GODOT_4:
            tscn_content = self._generate_godot4_scene(rig)
            script_content = self._generate_animator_script(rig, is_3d=False)
        elif format == ExportFormat.GODOT_4_3D:
            tscn_content = self._generate_godot4_3d_scene(rig)
            script_content = self._generate_animator_script(rig, is_3d=True)
        elif format == ExportFormat.GODOT_3:
            tscn_content = self._generate_godot3_scene(rig)
            script_content = self._generate_animator_script(rig, is_3d=False)
        elif format == ExportFormat.SPINE:
            spine_json = self._generate_spine_json(rig)
            spine_path = monster_dir / f"{rig.name}.json"
            spine_path.write_text(spine_json)
            logger.info(f"? Exported Spine rig to: {spine_path}")
            return str(spine_path)
        elif format == ExportFormat.PNG_SEQUENCE:
            # Just metadata, no scene
            metadata = self._generate_metadata(rig)
            meta_path = monster_dir / "rig_data.json"
            meta_path.write_text(json.dumps(metadata, indent=2))
            logger.info(f"? Exported PNGs to: {parts_dir}")
            return str(parts_dir)
        else:
            tscn_content = self._generate_godot4_scene(rig)
            script_content = self._generate_animator_script(rig, is_3d=False)
        
        tscn_path = monster_dir / f"{rig.name}.tscn"
        tscn_path.write_text(tscn_content)
        
        # Generate animator script
        script_path = monster_dir / f"{rig.name}_rig.gd"
        script_path.write_text(script_content)
        
        # Generate metadata
        metadata = self._generate_metadata(rig)
        meta_path = monster_dir / "rig_data.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        
        logger.info(f"? Exported rig to: {monster_dir}")
        return str(tscn_path)
    
    def _generate_godot4_scene(self, rig: MonsterRig) -> str:
        """Generate Godot 4.x scene file"""
        parts = rig.get_sorted_parts()
        
        lines = [
            f'[gd_scene load_steps={len(parts) + 2} format=3 uid="uid://{rig.name[:8]}"]',
            ''
        ]
        
        # External resources
        res_id = 1
        part_ids = {}
        
        for part in parts:
            lines.append(
                f'[ext_resource type="Texture2D" uid="uid://{part.name[:8]}" '
                f'path="res://monsters/{rig.name}/parts/{part.name}.png" id="{res_id}"]'
            )
            part_ids[part.name] = res_id
            res_id += 1
        
        # Script resource
        script_id = res_id
        lines.append(
            f'[ext_resource type="Script" path="res://monsters/{rig.name}/{rig.name}_rig.gd" id="{script_id}"]'
        )
        lines.append('')
        
        # Root node
        lines.append(f'[node name="{rig.name}" type="Node2D"]')
        lines.append(f'script = ExtResource("{script_id}")')
        lines.append('')
        
        # Part nodes
        for part in parts:
            # Determine parent path
            if part.parent and part.parent in rig.parts:
                parent_path = part.parent
            else:
                parent_path = "."
            
            lines.append(f'[node name="{part.name}" type="Sprite2D" parent="{parent_path}"]')
            lines.append(f'texture = ExtResource("{part_ids[part.name]}")')
            lines.append(f'centered = false')
            
            if part.bbox:
                lines.append(f'position = Vector2({part.bbox.x1}, {part.bbox.y1})')
            
            lines.append(f'offset = Vector2({-part.pivot[0]}, {-part.pivot[1]})')
            lines.append(f'z_index = {part.z_index}')
            lines.append('')
        
        return '\n'.join(lines)
    
    def _generate_godot3_scene(self, rig: MonsterRig) -> str:
        """Generate Godot 3.x scene file"""
        parts = rig.get_sorted_parts()
        
        lines = [
            f'[gd_scene load_steps={len(parts) + 2} format=2]',
            ''
        ]
        
        # Resources
        res_id = 1
        part_ids = {}
        
        for part in parts:
            lines.append(
                f'[ext_resource path="res://monsters/{rig.name}/parts/{part.name}.png" '
                f'type="Texture" id={res_id}]'
            )
            part_ids[part.name] = res_id
            res_id += 1
        
        script_id = res_id
        lines.append(f'[ext_resource path="res://monsters/{rig.name}/{rig.name}_rig.gd" type="Script" id={script_id}]')
        lines.append('')
        
        # Root
        lines.append(f'[node name="{rig.name}" type="Node2D"]')
        lines.append(f'script = ExtResource( {script_id} )')
        lines.append('')
        
        # Parts
        for part in parts:
            parent_path = part.parent if part.parent in rig.parts else "."
            
            lines.append(f'[node name="{part.name}" type="Sprite" parent="{parent_path}"]')
            lines.append(f'texture = ExtResource( {part_ids[part.name]} )')
            lines.append(f'centered = false')
            
            if part.bbox:
                lines.append(f'position = Vector2( {part.bbox.x1}, {part.bbox.y1} )')
            
            lines.append(f'offset = Vector2( {-part.pivot[0]}, {-part.pivot[1]} )')
            lines.append(f'z_index = {part.z_index}')
            lines.append('')
        
        return '\n'.join(lines)
    
    def _generate_animator_script(self, rig: MonsterRig, is_3d: bool = False) -> str:
        """Generate comprehensive animator GDScript"""
        parts = list(rig.parts.keys())
        safe_name = rig.name.replace("-", "_").replace(" ", "_")
        class_name = "".join(word.title() for word in safe_name.split("_")) + "Rig"
        
        extends_type = "Node3D" if is_3d else "Node2D"
        sprite_type = "Sprite3D" if is_3d else "Sprite2D"
        vector_type = "Vector3" if is_3d else "Vector2"
        
        script = f'''# ??????????????????????????????????????????????????????????????????????????????
# {rig.name.upper()} - VEILBREAKERS MONSTER RIG
# Generated by VeilbreakersRigger - The Ultimate Cutout Animation System
# {'3D Billboarded Sprites' if is_3d else '2D Sprites'}
# ??????????????????????????????????????????????????????????????????????????????

extends {extends_type}
class_name {class_name}

# ??????????????????????????????????????????????????????????????????????????????
# SIGNALS
# ??????????????????????????????????????????????????????????????????????????????
signal animation_started(anim_name: String)
signal animation_finished(anim_name: String)
signal hit_frame  ## Emitted at the impact point of attacks

# ??????????????????????????????????????????????????????????????????????????????
# EXPORTS - Tweak these in the Inspector
# ??????????????????????????????????????????????????????????????????????????????
@export_category("Idle Animation")
@export var idle_breathing: bool = true
@export var breathing_speed: float = 1.2
@export var breathing_intensity: float = 0.025
@export var idle_sway: bool = true
@export var sway_speed: float = 0.8
@export var sway_intensity: float = 2.0  # degrees

@export_category("Attack Animation")
@export var attack_windup_time: float = 0.2
@export var attack_strike_time: float = 0.08
@export var attack_recovery_time: float = 0.35
@export var attack_lunge_distance: float = {'1.5' if is_3d else '50.0'}
@export var attack_rotation: float = -5.0  # degrees during strike

@export_category("Hit Reaction")
@export var hit_knockback: float = {'0.8' if is_3d else '25.0'}
@export var hit_flash_duration: float = 0.08
@export var hit_recovery_time: float = 0.25
@export var hit_flash_color: Color = Color(3.0, 3.0, 3.0)

@export_category("Death Animation")
@export var death_fall_angle: float = 75.0
@export var death_fall_time: float = 0.6
@export var death_fade_time: float = 0.5
@export var death_bounce: bool = true

# ??????????????????????????????????????????????????????????????????????????????
# INTERNAL STATE
# ??????????????????????????????????????????????????????????????????????????????
var _base_transforms: Dictionary = {{}}
var _is_playing: bool = false
var _current_animation: String = ""
var _animation_tween: Tween = null
var _time: float = 0.0

# ??????????????????????????????????????????????????????????????????????????????
# NODE REFERENCES
# ??????????????????????????????????????????????????????????????????????????????
'''
        
        sprite_type = "Sprite3D" if is_3d else "Sprite2D"
        
        # Add @onready vars for each part
        for part in parts:
            safe_part = part.replace("-", "_").replace(" ", "_")
            script += f'@onready var _{safe_part}: {sprite_type} = ${part}\n'
        
        # Rest of the script adapts to 2D/3D
        script += f'''
# ??????????????????????????????????????????????????????????????????????????????
# LIFECYCLE
# ??????????????????????????????????????????????????????????????????????????????

func _ready() -> void:
    _cache_transforms()

func _process(delta: float) -> void:
    _time += delta
    
    if not _is_playing:
        _apply_idle_animation()

func _cache_transforms() -> void:
    """Store initial transforms for all parts"""
    for child in get_children():
        if child is {sprite_type}:
            _base_transforms[child.name] = {{
                "position": child.position,
                "rotation": child.rotation{'_degrees' if is_3d else ''},
                "scale": child.scale,
            }}

func _reset_to_base() -> void:
    """Reset all parts to their base transforms"""
    for child in get_children():
        if child is {sprite_type} and child.name in _base_transforms:
            var base = _base_transforms[child.name]
            child.position = base["position"]
            child.rotation{'_degrees' if is_3d else ''} = base["rotation"]
            child.scale = base["scale"]
            child.modulate = Color.WHITE

# ??????????????????????????????????????????????????????????????????????????????
# IDLE ANIMATION
# ??????????????????????????????????????????????????????????????????????????????

func _apply_idle_animation() -> void:
    if idle_breathing:
        _apply_breathing()
    if idle_sway:
        _apply_sway()

func _apply_breathing() -> void:
    var breath = sin(_time * breathing_speed * TAU) * breathing_intensity
    
    for child in get_children():
        if child is {sprite_type}:
            var n = child.name.to_lower()
            
            # Body parts expand/contract
            if "body" in n or "torso" in n or "chest" in n:
                child.scale.y = 1.0 + breath
                child.scale.x = 1.0 - breath * 0.3
            
            # Head bobs slightly
            elif "head" in n:
                if child.name in _base_transforms:
                    var base_pos = _base_transforms[child.name]["position"]
                    child.position.y = base_pos.y - breath * {'0.3' if is_3d else '8.0'}

func _apply_sway() -> void:
    var sway = sin(_time * sway_speed * TAU) * deg_to_rad(sway_intensity)
    rotation{'_degrees.y' if is_3d else ''} = {'rad_to_deg(sway)' if is_3d else 'sway'}

# ??????????????????????????????????????????????????????????????????????????????
# ANIMATION PLAYBACK
# ??????????????????????????????????????????????????????????????????????????????

func play_idle() -> void:
    """Return to idle state"""
    _stop_animation()
    _reset_to_base()
    _current_animation = "idle"

func play_attack() -> void:
    """Play attack animation with windup, strike, and recovery"""
    if _is_playing:
        return
    
    _start_animation("attack")
    
    var original_pos = position
    var original_rot = rotation
    
    _animation_tween = create_tween()
    _animation_tween.set_ease(Tween.EASE_OUT)
    
    # Windup - pull back and rotate
    _animation_tween.tween_property(self, "position:x", original_pos.x - attack_lunge_distance * 0.3, attack_windup_time)
    _animation_tween.parallel().tween_property(self, "rotation_degrees", attack_rotation * -0.5, attack_windup_time)
    
    # Strike - lunge forward
    _animation_tween.tween_property(self, "position:x", original_pos.x + attack_lunge_distance, attack_strike_time)\\
        .set_ease(Tween.EASE_IN)
    _animation_tween.parallel().tween_property(self, "rotation_degrees", attack_rotation, attack_strike_time)
    
    # Hit frame
    _animation_tween.tween_callback(_emit_hit_frame)
    
    # Recovery - return with bounce
    _animation_tween.tween_property(self, "position:x", original_pos.x, attack_recovery_time)\\
        .set_ease(Tween.EASE_OUT).set_trans(Tween.TRANS_ELASTIC)
    _animation_tween.parallel().tween_property(self, "rotation_degrees", 0.0, attack_recovery_time)\\
        .set_ease(Tween.EASE_OUT)
    
    _animation_tween.tween_callback(_finish_animation.bind("attack"))

func play_hurt() -> void:
    """Play hurt reaction"""
    if _current_animation == "death":
        return
    
    _start_animation("hurt")
    
    # Flash white
    modulate = hit_flash_color
    
    _animation_tween = create_tween()
    
    # Flash recovery
    _animation_tween.tween_property(self, "modulate", Color.WHITE, hit_flash_duration)
    
    # Knockback
    var original_x = position.x
    _animation_tween.parallel().tween_property(self, "position:x", original_x - hit_knockback, hit_flash_duration * 2)
    _animation_tween.tween_property(self, "position:x", original_x, hit_recovery_time)\\
        .set_ease(Tween.EASE_OUT).set_trans(Tween.TRANS_ELASTIC)
    
    _animation_tween.tween_callback(_finish_animation.bind("hurt"))

func play_death() -> void:
    """Play death animation"""
    _start_animation("death")
    
    _animation_tween = create_tween()
    
    # Fall over
    _animation_tween.tween_property(self, "rotation_degrees", death_fall_angle, death_fall_time)\\
        .set_ease(Tween.EASE_IN)
    _animation_tween.parallel().tween_property(self, "position:y", position.y + 30, death_fall_time)
    
    # Bounce if enabled
    if death_bounce:
        _animation_tween.tween_property(self, "rotation_degrees", death_fall_angle - 10, 0.1)
        _animation_tween.tween_property(self, "rotation_degrees", death_fall_angle, 0.1)
    
    # Fade
    _animation_tween.tween_property(self, "modulate:a", 0.3, death_fade_time)
    
    _animation_tween.tween_callback(_finish_animation.bind("death"))

func play_special(intensity: float = 1.0) -> void:
    """Play special/charge animation"""
    if _is_playing:
        return
    
    _start_animation("special")
    
    _animation_tween = create_tween()
    
    # Pulse effect
    _animation_tween.tween_property(self, "scale", Vector2(1.0 + 0.1 * intensity, 1.0 + 0.1 * intensity), 0.15)
    _animation_tween.tween_property(self, "scale", Vector2(0.95, 0.95), 0.1)
    _animation_tween.tween_property(self, "scale", Vector2(1.0 + 0.15 * intensity, 1.0 + 0.15 * intensity), 0.2)
    
    # Flash
    _animation_tween.tween_callback(func(): modulate = Color(1.5, 1.2, 1.0))
    
    _animation_tween.tween_property(self, "scale", Vector2.ONE, 0.15)
    _animation_tween.parallel().tween_property(self, "modulate", Color.WHITE, 0.15)
    
    _animation_tween.tween_callback(_finish_animation.bind("special"))

# ??????????????????????????????????????????????????????????????????????????????
# ANIMATION UTILITIES
# ??????????????????????????????????????????????????????????????????????????????

func _start_animation(anim_name: String) -> void:
    _stop_animation()
    _is_playing = true
    _current_animation = anim_name
    animation_started.emit(anim_name)

func _stop_animation() -> void:
    if _animation_tween and _animation_tween.is_valid():
        _animation_tween.kill()
    _is_playing = false

func _finish_animation(anim_name: String) -> void:
    _is_playing = false
    animation_finished.emit(anim_name)

func _emit_hit_frame() -> void:
    hit_frame.emit()

func is_animating() -> bool:
    return _is_playing

func get_current_animation() -> String:
    return _current_animation

# ??????????????????????????????????????????????????????????????????????????????
# PART ACCESS
# ??????????????????????????????????????????????????????????????????????????????

func get_part(part_name: String) -> Sprite2D:
    """Get a specific body part sprite"""
    return get_node_or_null(part_name) as Sprite2D

func set_part_visible(part_name: String, visible: bool) -> void:
    """Show/hide a specific part"""
    var part = get_part(part_name)
    if part:
        part.visible = visible

func set_part_modulate(part_name: String, color: Color) -> void:
    """Set the color modulate of a specific part"""
    var part = get_part(part_name)
    if part:
        part.modulate = color
'''
        
        return script
    
    def _generate_godot4_3d_scene(self, rig: MonsterRig) -> str:
        """Generate Godot 4.x 3D scene with billboarded Sprite3D nodes"""
        parts = rig.get_sorted_parts()
        
        lines = [
            f'[gd_scene load_steps={len(parts) + 2} format=3 uid="uid://{rig.name[:8]}3d"]',
            ''
        ]
        
        # External resources
        res_id = 1
        part_ids = {}
        
        for part in parts:
            lines.append(
                f'[ext_resource type="Texture2D" uid="uid://{part.name[:8]}3d" '
                f'path="res://monsters/{rig.name}/parts/{part.name}.png" id="{res_id}"]'
            )
            part_ids[part.name] = res_id
            res_id += 1
        
        # Script resource
        script_id = res_id
        lines.append(
            f'[ext_resource type="Script" path="res://monsters/{rig.name}/{rig.name}_rig.gd" id="{script_id}"]'
        )
        lines.append('')
        
        # Root Node3D
        lines.append(f'[node name="{rig.name}" type="Node3D"]')
        lines.append(f'script = ExtResource("{script_id}")')
        lines.append('')
        
        # Convert pixels to 3D units (100 pixels = 1 unit)
        scale_factor = 0.01
        
        # Part Sprite3D nodes
        for part in parts:
            # Determine parent path
            if part.parent and part.parent in rig.parts:
                parent_path = part.parent
            else:
                parent_path = "."
            
            lines.append(f'[node name="{part.name}" type="Sprite3D" parent="{parent_path}"]')
            lines.append(f'texture = ExtResource("{part_ids[part.name]}")')
            lines.append(f'billboard = 1')  # Fixed Y-axis billboard
            lines.append(f'transparent = true')
            lines.append(f'alpha_cut = 1')  # Opaque pre-pass for proper depth
            lines.append(f'texture_filter = 0')  # Nearest for pixel art
            
            if part.bbox:
                # Convert 2D position to 3D (X stays X, Y becomes Y height, Z=0)
                x_pos = part.bbox.x1 * scale_factor
                y_pos = -part.bbox.y1 * scale_factor  # Invert Y for 3D
                # Use z_index for slight depth offset
                z_pos = part.z_index * 0.01
                lines.append(f'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {x_pos:.4f}, {y_pos:.4f}, {z_pos:.4f})')
            
            lines.append(f'centered = false')
            lines.append(f'offset = Vector2({-part.pivot[0]}, {-part.pivot[1]})')
            lines.append(f'pixel_size = {scale_factor}')  # Scale pixels to world units
            lines.append('')
        
        return '\n'.join(lines)
    
    def _generate_spine_json(self, rig: MonsterRig) -> str:
        """Generate Spine JSON format for compatibility with other tools"""
        parts = rig.get_sorted_parts()
        
        # Helper to convert numpy types to Python native
        def to_native(val):
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            return val
        
        # Build skeleton structure
        bones = [{"name": "root"}]
        slots = []
        skins = {"default": {}}
        
        # Create bones from parts
        for i, part in enumerate(parts):
            bone = {
                "name": part.name,
                "parent": part.parent if part.parent else "root",
                "x": to_native(part.pivot_world[0]) if part.pivot_world else 0,
                "y": to_native(-part.pivot_world[1]) if part.pivot_world else 0,  # Spine uses different Y
            }
            bones.append(bone)
            
            # Create slot for this bone
            slot = {
                "name": part.name,
                "bone": part.name,
                "attachment": part.name
            }
            slots.append(slot)
            
            # Create skin attachment
            if part.bbox:
                skins["default"][part.name] = {
                    part.name: {
                        "type": "region",
                        "x": 0,
                        "y": 0,
                        "width": to_native(part.bbox.width),
                        "height": to_native(part.bbox.height),
                    }
                }
        
        # Build complete Spine structure
        spine_data = {
            "skeleton": {
                "hash": rig.name,
                "spine": "4.0.0",
                "x": 0,
                "y": 0,
                "width": int(rig.original_image.shape[1]) if rig.original_image is not None else 512,
                "height": int(rig.original_image.shape[0]) if rig.original_image is not None else 512,
                "images": f"./parts/",
            },
            "bones": bones,
            "slots": slots,
            "skins": [
                {
                    "name": "default",
                    "attachments": skins["default"]
                }
            ],
            "animations": {
                "idle": {
                    "bones": {}  # Can be filled with keyframe data
                }
            }
        }
        
        return json.dumps(spine_data, indent=2)
    
    def _generate_metadata(self, rig: MonsterRig) -> dict:
        """Generate comprehensive metadata"""
        from datetime import datetime
        
        # Helper to convert numpy types to Python native
        def to_native(val):
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            elif isinstance(val, (list, tuple)):
                return [to_native(v) for v in val]
            return val
        
        parts_data = {}
        for name, part in rig.parts.items():
            parts_data[name] = {
                "bbox": to_native(part.bbox.to_xywh()) if part.bbox else None,
                "pivot": to_native(list(part.pivot)),
                "pivot_world": to_native(list(part.pivot_world)),
                "z_index": int(part.z_index),
                "parent": part.parent,
                "children": part.children,
                "confidence": float(part.confidence) if part.confidence else None
            }
        
        return {
            "name": rig.name,
            "template": rig.template,
            "parts": parts_data,
            "hierarchy": rig.hierarchy,
            "root_parts": rig.root_parts,
            "generated_at": datetime.now().isoformat(),
            "generator": "VeilbreakersRigger v2.0"
        }

# =============================================================================
# MAIN RIGGER CLASS
# =============================================================================

class VeilbreakersRigger:
    """
    ????????????????????????????????????????????????????????????????????????????
    ?                    THE ULTIMATE MONSTER RIGGER                           ?
    ????????????????????????????????????????????????????????????????????????????
    ?                                                                          ?
    ?  FEATURES:                                                               ?
    ?  ? Auto-detect body parts from text prompts (AI-powered)                 ?
    ?  ? Manual click-to-segment for precise control                           ?
    ?  ? Positive/negative point refinement                                    ?
    ?  ? High-quality inpainting for overlapping parts                         ?
    ?  ? Automatic pivot point calculation                                     ?
    ?  ? Hierarchical bone structure                                           ?
    ?  ? Godot 3/4 export with animation scripts                               ?
    ?                                                                          ?
    ?  USAGE:                                                                  ?
    ?  >>> rigger = VeilbreakersRigger()                                       ?
    ?  >>> rigger.load_image("monster.png")                                    ?
    ?  >>>                                                                     ?
    ?  >>> # Auto-detect parts                                                 ?
    ?  >>> rigger.auto_detect("head . body . arms . legs . tail")              ?
    ?  >>>                                                                     ?
    ?  >>> # Or use preset                                                     ?
    ?  >>> rigger.auto_detect_preset("quadruped")                              ?
    ?  >>>                                                                     ?
    ?  >>> # Manual segmentation for overlapping parts                         ?
    ?  >>> rigger.click_segment(150, 200, "arm_left")                          ?
    ?  >>>                                                                     ?
    ?  >>> # Refine with additional points                                     ?
    ?  >>> rigger.refine_add(160, 210)                                         ?
    ?  >>> rigger.refine_subtract(140, 250)                                    ?
    ?  >>> rigger.confirm_selection("arm_left", z_index=3)                     ?
    ?  >>>                                                                     ?
    ?  >>> # Export                                                            ?
    ?  >>> rigger.export("shadow_wolf")                                        ?
    ?                                                                          ?
    ????????????????????????????????????????????????????????????????????????????
    """
    
    def __init__(self,
                 output_dir: str = "./output",
                 sam_size: str = "large",
                 device: str = "auto",
                 use_fallback: bool = True):
        """
        Initialize the rigger
        
        Args:
            output_dir: Directory for exported rigs
            sam_size: SAM 2 model size ("tiny", "small", "base", "large")
            device: Compute device ("auto", "cuda", "cpu", "mps")
            use_fallback: Use OpenCV fallback if SAM2 unavailable
        """
        self.output_dir = Path(output_dir)
        self.use_fallback = use_fallback
        
        # Initialize engines
        try:
            self.segmenter = GroundedSAM2Engine(sam_size=sam_size, device=device)
        except Exception as e:
            if use_fallback:
                logger.warning(f"SAM2 unavailable ({e}), using OpenCV fallback")
                self.segmenter = OpenCVSegmentationEngine()
            else:
                raise
        
        self.inpainter = InpaintingEngine()
        self.extractor = PartExtractor(self.inpainter)
        self.exporter = GodotExporter(output_dir)
        
        # Current state
        self.current_rig: Optional[MonsterRig] = None
        self.current_mask: Optional[np.ndarray] = None
        self.positive_points: List[Point] = []
        self.negative_points: List[Point] = []
        
        logger.info("VeilbreakersRigger initialized ?")
    
    # ?????????????????????????????????????????????????????????????????????????
    # IMAGE LOADING
    # ?????????????????????????????????????????????????????????????????????????
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load a monster image for rigging
        
        Args:
            image_path: Path to the image file
            
        Returns:
            The loaded RGB image as numpy array
        """
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        
        name = Path(image_path).stem
        
        self.current_rig = MonsterRig(
            name=name,
            original_image=image_array,
            working_image=image_array.copy()
        )
        
        self.segmenter.set_image(image_array)
        self._clear_selection()
        
        logger.info(f"Loaded image: {image_path} ({image_array.shape[1]}x{image_array.shape[0]})")
        return image_array
    
    def load_image_array(self, image: np.ndarray, name: str = "monster") -> np.ndarray:
        """Load from numpy array"""
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        self.current_rig = MonsterRig(
            name=name,
            original_image=image,
            working_image=image.copy()
        )
        
        self.segmenter.set_image(image)
        self._clear_selection()
        
        return image
    
    # ?????????????????????????????????????????????????????????????????????????
    # AUTO-DETECTION
    # ?????????????????????????????????????????????????????????????????????????
    
    def auto_detect(self,
                    text_prompt: str,
                    box_threshold: float = 0.25,
                    text_threshold: float = 0.25,
                    extract_parts: bool = True,
                    inpaint_quality: InpaintQuality = InpaintQuality.STANDARD) -> List[BodyPart]:
        """
        Auto-detect and segment body parts from text prompt
        
        Args:
            text_prompt: Parts to detect, separated by " . "
                         Example: "head . body . arms . legs . tail"
            box_threshold: Detection confidence threshold (0-1)
            text_threshold: Text matching threshold (0-1)
            extract_parts: Whether to extract parts immediately
            inpaint_quality: Quality level for hole filling
            
        Returns:
            List of detected BodyPart objects
        """
        if self.current_rig is None:
            raise ValueError("No image loaded")
        
        logger.info(f"Auto-detecting: {text_prompt}")
        
        detections = self.segmenter.auto_detect(
            text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        if not detections:
            return []

        parts = []

        # IMPORTANT: Save original image - extract ALL parts from original, not inpainted
        original_image = self.current_rig.original_image.copy()

        # Combine all masks for final inpainting
        combined_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)

        for i, (name, mask, confidence) in enumerate(detections):
            if extract_parts:
                # Extract from ORIGINAL image, not working image
                part, _ = self.extractor.extract(
                    original_image,  # Always use original
                    mask,
                    name,
                    z_index=i,
                    inpaint=False  # Don't inpaint yet
                )
                part.confidence = confidence
                self.current_rig.add_part(part)

                # Accumulate mask for final inpainting
                combined_mask = np.maximum(combined_mask, mask)
            else:
                part = BodyPart(name=name, mask=mask, confidence=confidence, z_index=i)

            parts.append(part)

        # Inpaint all extracted regions at once (for visualization only)
        if extract_parts and np.any(combined_mask > 0):
            self.current_rig.working_image = self.inpainter.inpaint(
                original_image,
                combined_mask,
                quality=inpaint_quality
            )
            self.segmenter.set_image(self.current_rig.working_image)

        return parts
    
    def auto_detect_preset(self,
                           preset_name: str,
                           extract_parts: bool = True,
                           inpaint_quality: InpaintQuality = InpaintQuality.STANDARD) -> List[BodyPart]:
        """
        Auto-detect using a predefined body template
        
        Args:
            preset_name: Name of preset ("quadruped", "humanoid", "winged", etc.)
            extract_parts: Whether to extract parts immediately
            inpaint_quality: Quality level for hole filling
        """
        if preset_name not in BODY_TEMPLATES:
            available = ", ".join(BODY_TEMPLATES.keys())
            raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
        
        template = BODY_TEMPLATES[preset_name]
        
        if not template["prompt"]:
            raise ValueError("Custom preset requires manual prompt")
        
        logger.info(f"Using preset: {template['name']}")
        
        parts = self.auto_detect(
            template["prompt"],
            extract_parts=extract_parts,
            inpaint_quality=inpaint_quality
        )
        
        # Apply z-order and pivots from template
        z_order = template.get("z_order", [])
        pivots = template.get("pivots", {})
        
        for part in parts:
            # Find z-index
            for i, name in enumerate(z_order):
                if name in part.name:
                    part.z_index = i
                    break
            
            # Find pivot type
            for name, pivot_type in pivots.items():
                if name in part.name:
                    part.pivot = part.calculate_pivot(pivot_type)
                    break
        
        # Set template info
        if self.current_rig:
            self.current_rig.template = preset_name
            self.current_rig.hierarchy = template.get("hierarchy", {})

        return parts

    def smart_detect(self,
                     text_prompt: str = None,
                     use_florence: bool = True,
                     box_threshold: float = 0.25,
                     extract_parts: bool = True,
                     inpaint_quality: InpaintQuality = InpaintQuality.STANDARD) -> List[BodyPart]:
        """
        BEST detection method - uses Florence-2 unified vision model.

        This is the PRIMARY method that should be used. It:
        1. Uses Florence-2 to detect AND locate all parts (no text prompt needed)
        2. Falls back to text prompt with Grounding DINO if needed
        3. Falls back to SAM Everything mode as last resort

        Args:
            text_prompt: Optional text prompt as fallback
            use_florence: Whether to use Florence-2 (recommended: True)
            box_threshold: Detection confidence threshold
            extract_parts: Whether to extract parts immediately
            inpaint_quality: Quality level for hole filling

        Returns:
            List of detected BodyPart objects
        """
        if self.current_rig is None:
            raise ValueError("No image loaded")

        logger.info(f"Smart-detecting (Florence-2={use_florence}, prompt={text_prompt})")

        # Use the segmenter's smart_detect method
        detections = self.segmenter.smart_detect(
            text_prompt=text_prompt,
            use_florence=use_florence,
            box_threshold=box_threshold
        )

        if not detections:
            return []

        parts = []

        # IMPORTANT: Save original image - extract ALL parts from original, not inpainted
        original_image = self.current_rig.original_image.copy()

        # Combine all masks for final inpainting
        combined_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)

        for i, (name, mask, confidence) in enumerate(detections):
            if extract_parts:
                # Extract from ORIGINAL image, not working image
                part, _ = self.extractor.extract(
                    original_image,
                    mask,
                    name,
                    z_index=i,
                    inpaint=False
                )
                part.confidence = confidence
                self.current_rig.add_part(part)

                # Accumulate mask for final inpainting
                combined_mask = np.maximum(combined_mask, mask)
            else:
                part = BodyPart(name=name, mask=mask, confidence=confidence, z_index=i)

            parts.append(part)

        # Inpaint all extracted regions at once
        if extract_parts and np.any(combined_mask > 0):
            self.current_rig.working_image = self.inpainter.inpaint(
                original_image,
                combined_mask,
                quality=inpaint_quality
            )
            self.segmenter.set_image(self.current_rig.working_image)

        return parts

    def segment_everything(self,
                           min_mask_area: int = 1000) -> List[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        """
        Segment ALL regions in the image automatically.
        User can then select which segments to keep and name them.

        Args:
            min_mask_area: Minimum pixel area for a segment

        Returns:
            List of (mask, score, bbox) tuples
        """
        if self.current_rig is None:
            raise ValueError("No image loaded")

        # Get all segments
        segments = self.segmenter.segment_everything(min_mask_area=min_mask_area)

        # Store for later selection
        self._all_segments = segments

        logger.info(f"Found {len(segments)} segments. Use select_segment() to pick one.")
        return segments

    def select_segment(self, index: int) -> np.ndarray:
        """
        Select a segment from the segment_everything results.

        Args:
            index: Index of the segment to select (0-based)

        Returns:
            The selected mask
        """
        if not hasattr(self, '_all_segments') or not self._all_segments:
            raise ValueError("No segments available. Call segment_everything() first.")

        if index < 0 or index >= len(self._all_segments):
            raise ValueError(f"Invalid index {index}. Available: 0-{len(self._all_segments)-1}")

        mask, score, bbox = self._all_segments[index]
        self.current_mask = mask
        if self.current_rig:
            self.current_rig.current_mask = mask

        logger.info(f"Selected segment {index} (score: {score:.2f}, area: {(mask > 0).sum()} px)")
        return mask

    def box_segment(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """
        Segment within a drawn box. SAM will find the best object in the box.

        Args:
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner

        Returns:
            Binary mask of the segmented region
        """
        if self.current_rig is None:
            raise ValueError("No image loaded")

        self._clear_selection()

        box = BoundingBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self.current_mask = self.segmenter.segment_box(box)
        self.current_rig.current_mask = self.current_mask

        logger.info(f"Box segment: ({x1},{y1}) to ({x2},{y2})")
        return self.current_mask

    def get_segment_preview(self, index: int) -> np.ndarray:
        """
        Get a preview image with segment highlighted.

        Args:
            index: Segment index

        Returns:
            RGB image with segment highlighted
        """
        if not hasattr(self, '_all_segments') or not self._all_segments:
            raise ValueError("No segments available")

        if self.current_rig is None:
            raise ValueError("No image loaded")

        mask, _, _ = self._all_segments[index]
        vis = self.current_rig.original_image.copy()

        # Create colored overlay
        overlay = np.zeros_like(vis)
        overlay[mask > 0] = [50, 255, 50]  # Green

        # Blend
        vis = (vis.astype(float) * 0.6 + overlay.astype(float) * 0.4).astype(np.uint8)

        return vis

    # ?????????????????????????????????????????????????????????????????????????
    # MANUAL SEGMENTATION
    # ?????????????????????????????????????????????????????????????????????????

    def click_segment(self, x: int, y: int) -> np.ndarray:
        """
        Start a new segmentation at a click point
        
        Args:
            x, y: Click coordinates
            
        Returns:
            Binary mask of the segmented region
        """
        if self.current_rig is None:
            raise ValueError("No image loaded")
        
        self._clear_selection()
        
        point = Point(x, y, label=1)
        self.positive_points.append(point)
        
        self.current_mask = self.segmenter.segment_point(point)
        self.current_rig.current_mask = self.current_mask
        
        logger.info(f"Segmented at ({x}, {y})")
        return self.current_mask
    
    def refine_add(self, x: int, y: int) -> np.ndarray:
        """
        Add to current selection with a positive point
        
        Args:
            x, y: Click coordinates to add
            
        Returns:
            Updated mask
        """
        if self.current_mask is None:
            return self.click_segment(x, y)
        
        point = Point(x, y, label=1)
        self.positive_points.append(point)
        
        self.current_mask = self.segmenter.segment_points(
            self.positive_points,
            self.negative_points
        )
        self.current_rig.current_mask = self.current_mask
        
        logger.info(f"Added point at ({x}, {y})")
        return self.current_mask
    
    def refine_subtract(self, x: int, y: int) -> np.ndarray:
        """
        Remove from current selection with a negative point
        
        Args:
            x, y: Click coordinates to exclude
            
        Returns:
            Updated mask
        """
        if self.current_mask is None:
            raise ValueError("No current selection to refine")
        
        point = Point(x, y, label=0)
        self.negative_points.append(point)
        
        self.current_mask = self.segmenter.segment_points(
            self.positive_points,
            self.negative_points
        )
        self.current_rig.current_mask = self.current_mask
        
        logger.info(f"Subtracted point at ({x}, {y})")
        return self.current_mask

    # NOTE: box_segment is defined earlier in this class (line ~2604) with proper min/max coordinate handling

    def confirm_selection(self,
                          name: str,
                          z_index: int = 0,
                          parent: str = "",
                          pivot_type: str = "center",
                          inpaint: bool = True,
                          inpaint_quality: InpaintQuality = InpaintQuality.STANDARD) -> BodyPart:
        """
        Confirm current selection and extract as a body part
        
        Args:
            name: Name for this part
            z_index: Layer order (higher = in front)
            parent: Parent part name for hierarchy
            pivot_type: Pivot point type
            inpaint: Whether to fill the hole
            inpaint_quality: Quality level for hole filling
            
        Returns:
            The extracted BodyPart
        """
        if self.current_mask is None:
            raise ValueError("No current selection")

        # Extract from ORIGINAL image to get correct colors
        part, _ = self.extractor.extract(
            self.current_rig.original_image,  # Use original, not working
            self.current_mask,
            name,
            pivot_type=pivot_type,
            z_index=z_index,
            parent=parent,
            inpaint=False  # Don't inpaint during extraction
        )

        # Store the points used
        part.positive_points = self.positive_points.copy()
        part.negative_points = self.negative_points.copy()

        self.current_rig.add_part(part)

        # Update working image with inpainting for visualization
        if inpaint:
            self.current_rig.working_image = self.inpainter.inpaint(
                self.current_rig.working_image,
                self.current_mask,
                quality=inpaint_quality
            )
            self.segmenter.set_image(self.current_rig.working_image)

        self._clear_selection()

        logger.info(f"Added part: {name} (z={z_index})")
        return part

    # Alias for UI compatibility
    def add_part(self, name: str, z_index: int = 0,
                 inpaint_quality: InpaintQuality = InpaintQuality.STANDARD) -> BodyPart:
        """Alias for confirm_selection with common defaults"""
        return self.confirm_selection(
            name=name,
            z_index=z_index,
            inpaint=True,
            inpaint_quality=inpaint_quality
        )

    def _clear_selection(self) -> None:
        """Clear current selection state"""
        self.current_mask = None
        self.positive_points = []
        self.negative_points = []
        if self.current_rig:
            self.current_rig.current_mask = None
    
    # ?????????????????????????????????????????????????????????????????????????
    # PART MANAGEMENT
    # ?????????????????????????????????????????????????????????????????????????
    
    def get_parts(self) -> List[BodyPart]:
        """Get all extracted parts"""
        if self.current_rig is None:
            return []
        return list(self.current_rig.parts.values())
    
    def get_part(self, name: str) -> Optional[BodyPart]:
        """Get a specific part by name"""
        if self.current_rig is None:
            return None
        return self.current_rig.get_part(name)
    
    def remove_part(self, name: str) -> None:
        """Remove a part from the rig"""
        if self.current_rig:
            self.current_rig.remove_part(name)
            logger.info(f"Removed part: {name}")
    
    def set_part_z_index(self, name: str, z_index: int) -> None:
        """Change a part's z-index"""
        part = self.get_part(name)
        if part:
            part.z_index = z_index
    
    def set_part_parent(self, name: str, parent: str) -> None:
        """Set a part's parent"""
        part = self.get_part(name)
        if part:
            # Remove from old parent
            if part.parent and part.parent in self.current_rig.parts:
                old_parent = self.current_rig.parts[part.parent]
                if name in old_parent.children:
                    old_parent.children.remove(name)
            
            # Add to new parent
            part.parent = parent
            if parent and parent in self.current_rig.parts:
                if name not in self.current_rig.parts[parent].children:
                    self.current_rig.parts[parent].children.append(name)
    
    def set_part_pivot(self, name: str, pivot_type: str) -> None:
        """Change a part's pivot point"""
        part = self.get_part(name)
        if part:
            part.pivot = part.calculate_pivot(pivot_type)
    
    # ?????????????????????????????????????????????????????????????????????????
    # VISUALIZATION
    # ?????????????????????????????????????????????????????????????????????????
    
    def get_visualization(self, 
                          show_masks: bool = True,
                          show_points: bool = True,
                          show_pivots: bool = True,
                          alpha: float = 0.4) -> np.ndarray:
        """
        Get a visualization of the current state
        
        Args:
            show_masks: Overlay extracted part masks
            show_points: Show positive/negative points
            show_pivots: Show pivot points
            alpha: Mask overlay transparency
            
        Returns:
            Visualization image
        """
        if self.current_rig is None:
            raise ValueError("No image loaded")
        
        vis = self.current_rig.working_image.copy()
        
        # Overlay current selection
        if self.current_mask is not None:
            overlay = np.zeros_like(vis)
            overlay[:, :, 0] = self.current_mask  # Red
            overlay[:, :, 1] = self.current_mask // 3  # Some green
            vis = (vis * (1 - alpha) + overlay * alpha).astype(np.uint8)
        
        # Draw points
        if show_points:
            vis = self._draw_points(vis)
        
        # Draw pivots
        if show_pivots:
            vis = self._draw_pivots(vis)
        
        return vis
    
    def _draw_points(self, image: np.ndarray) -> np.ndarray:
        """Draw positive and negative points"""
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # Positive points (green circles)
        for point in self.positive_points:
            draw.ellipse(
                [point.x - 6, point.y - 6, point.x + 6, point.y + 6],
                fill=(0, 255, 0),
                outline=(255, 255, 255),
                width=2
            )
        
        # Negative points (red circles)
        for point in self.negative_points:
            draw.ellipse(
                [point.x - 6, point.y - 6, point.x + 6, point.y + 6],
                fill=(255, 0, 0),
                outline=(255, 255, 255),
                width=2
            )
        
        return np.array(pil_image)
    
    def _draw_pivots(self, image: np.ndarray) -> np.ndarray:
        """Draw pivot points for extracted parts"""
        if not self.current_rig:
            return image
        
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        for part in self.current_rig.parts.values():
            px, py = part.pivot_world
            
            # Draw crosshair
            size = 8
            draw.line([px - size, py, px + size, py], fill=(255, 255, 0), width=2)
            draw.line([px, py - size, px, py + size], fill=(255, 255, 0), width=2)
            
            # Draw circle
            draw.ellipse(
                [px - 4, py - 4, px + 4, py + 4],
                fill=(255, 255, 0),
                outline=(0, 0, 0),
                width=1
            )
        
        return np.array(pil_image)
    
    # ?????????????????????????????????????????????????????????????????????????
    # EXPORT
    # ?????????????????????????????????????????????????????????????????????????
    
    def export(self, 
               name: str = None,
               format: ExportFormat = ExportFormat.GODOT_4) -> str:
        """
        Export the rig to game-ready files
        
        Args:
            name: Monster name (uses image name if not specified)
            format: Export format (GODOT_4, GODOT_3, etc.)
            
        Returns:
            Path to the main scene file
        """
        if self.current_rig is None:
            raise ValueError("No rig to export")
        
        if not self.current_rig.parts:
            raise ValueError("No parts extracted")
        
        if name:
            self.current_rig.name = name
        
        return self.exporter.export(self.current_rig, format)
    
    # ?????????????????????????????????????????????????????????????????????????
    # UTILITIES
    # ?????????????????????????????????????????????????????????????????????????
    
    @staticmethod
    def get_presets() -> Dict[str, dict]:
        """Get available body templates"""
        return BODY_TEMPLATES
    
    @staticmethod
    def get_preset_names() -> List[str]:
        """Get list of preset names"""
        return list(BODY_TEMPLATES.keys())
    
    @staticmethod
    def get_preset_info(preset_name: str) -> Optional[dict]:
        """Get info about a specific preset"""
        return BODY_TEMPLATES.get(preset_name)
    
    def update_part(self, name: str, 
                    z_index: int = None,
                    parent: str = None,
                    pivot_type: str = None) -> Optional[BodyPart]:
        """
        Update properties of an existing part
        
        Args:
            name: Part name to update
            z_index: New z-index (optional)
            parent: New parent name (optional)
            pivot_type: New pivot type (optional)
            
        Returns:
            Updated BodyPart or None if not found
        """
        if not self.current_rig or name not in self.current_rig.parts:
            return None
        
        part = self.current_rig.parts[name]
        
        if z_index is not None:
            part.z_index = z_index
        
        if parent is not None:
            # Remove from old parent's children
            if part.parent and part.parent in self.current_rig.parts:
                old_parent = self.current_rig.parts[part.parent]
                if name in old_parent.children:
                    old_parent.children.remove(name)
            
            # Update parent
            part.parent = parent
            
            # Add to new parent's children
            if parent and parent in self.current_rig.parts:
                self.current_rig.parts[parent].children.append(name)
                if name in self.current_rig.root_parts:
                    self.current_rig.root_parts.remove(name)
            elif not parent:
                if name not in self.current_rig.root_parts:
                    self.current_rig.root_parts.append(name)
        
        if pivot_type is not None and part.bbox:
            part.pivot = part.calculate_pivot(pivot_type)
            part.pivot_world = (
                part.bbox.x1 + part.pivot[0],
                part.bbox.y1 + part.pivot[1]
            )
        
        logger.info(f"Updated part: {name}")
        return part
    
    def get_working_image(self) -> Optional[np.ndarray]:
        """Get the current working image (with holes filled)"""
        if self.current_rig:
            return self.current_rig.working_image
        return None
    
    def get_original_image(self) -> Optional[np.ndarray]:
        """Get the original image"""
        if self.current_rig:
            return self.current_rig.original_image
        return None
    
    def reset(self) -> None:
        """Reset to original image state"""
        if self.current_rig:
            self.current_rig.working_image = self.current_rig.original_image.copy()
            self.current_rig.parts = {}
            self.current_rig.root_parts = []
            self.segmenter.set_image(self.current_rig.original_image)
        self._clear_selection()
        logger.info("Reset to original state")

# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VEILBREAKERS Monster Rigger - The Ultimate Cutout Rig System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Auto-rig with preset
  python veilbreakers_rigger.py monster.png --preset quadruped
  
  # Auto-rig with custom prompt  
  python veilbreakers_rigger.py monster.png --prompt "head . body . wings . tail"
  
  # High-quality inpainting
  python veilbreakers_rigger.py monster.png --preset humanoid --quality high
  
  # Launch interactive GUI
  python veilbreakers_rigger.py --gui

PRESETS:
  quadruped  - 4-legged (wolf, cat, dragon)
  humanoid   - 2 arms, 2 legs (golem, demon)
  winged     - With wings (dragon, demon, bird)
  serpent    - Snake-like (serpent, wyrm)
  spider     - 8-legged (spider, beetle)
  floating   - Floating (ghost, beholder)
        """
    )
    
    parser.add_argument("image", nargs="?", help="Path to monster image")
    parser.add_argument("--preset", choices=list(BODY_TEMPLATES.keys()),
                        help="Body template preset")
    parser.add_argument("--prompt", help="Custom detection prompt")
    parser.add_argument("--output", "-o", default="./output",
                        help="Output directory")
    parser.add_argument("--name", "-n", help="Monster name")
    parser.add_argument("--quality", choices=["fast", "standard", "high", "ultra"],
                        default="standard", help="Inpainting quality")
    parser.add_argument("--model", choices=["tiny", "small", "base", "large"],
                        default="large", help="SAM model size")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--list-presets", action="store_true", 
                        help="List available presets")
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("\nAvailable presets:")
        for name, template in BODY_TEMPLATES.items():
            print(f"  {name:12} - {template['name']}")
            if template['prompt']:
                print(f"               Parts: {template['prompt']}")
        return
    
    if args.gui:
        print("Launching GUI...")
        try:
            from veilbreakers_rigger_ui import launch_ui
            launch_ui()
        except ImportError:
            print("GUI not available. Install gradio: pip install gradio")
        return
    
    if not args.image:
        parser.print_help()
        return
    
    # Quality mapping
    quality_map = {
        "fast": InpaintQuality.FAST,
        "standard": InpaintQuality.STANDARD,
        "high": InpaintQuality.HIGH,
        "ultra": InpaintQuality.ULTRA
    }
    
    # Run rigging
    rigger = VeilbreakersRigger(
        output_dir=args.output,
        sam_size=args.model
    )
    
    rigger.load_image(args.image)
    
    quality = quality_map[args.quality]
    
    if args.preset:
        rigger.auto_detect_preset(args.preset, inpaint_quality=quality)
    elif args.prompt:
        rigger.auto_detect(args.prompt, inpaint_quality=quality)
    else:
        print("Specify --preset or --prompt for auto-detection")
        return
    
    output_path = rigger.export(name=args.name)
    print(f"\n? Exported to: {output_path}")


if __name__ == "__main__":
    main()
