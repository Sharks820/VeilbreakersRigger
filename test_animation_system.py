#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMPREHENSIVE TEST SUITE                                  ║
║                                                                              ║
║   Tests all components of the VEILBREAKERS Animation System                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import traceback
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test counters
passed = 0
failed = 0
warnings = []
errors = []

def test(name: str, condition: bool, error_msg: str = ""):
    """Run a single test"""
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}")
        if error_msg:
            print(f"     → {error_msg}")
        failed += 1
        errors.append(f"{name}: {error_msg}")

def section(name: str):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

def run_tests():
    """Run all tests"""
    global passed, failed, warnings, errors
    
    print("\n" + "="*70)
    print("  VEILBREAKERS ANIMATION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    # =========================================================================
    # SECTION 1: CORE IMPORTS
    # =========================================================================
    section("1. CORE IMPORTS")
    
    try:
        from animation_engine import (
            PartType, CreatureArchetype, Bone, Slot, IKConstraint,
            PhysicsConstraint, Animation, CreatureRig, PartClassifier,
            BoneChainGenerator, AnimationGenerator, PHYSICS_PRESETS
        )
        test("animation_engine imports", True)
    except Exception as e:
        test("animation_engine imports", False, str(e))
        return  # Can't continue without core
    
    try:
        from animation_templates import AnimationTemplates
        test("animation_templates imports", True)
    except Exception as e:
        test("animation_templates imports", False, str(e))
    
    try:
        from spine_rig_builder import SpineRigBuilder, ARCHETYPE_CONFIGS
        test("spine_rig_builder imports", True)
    except Exception as e:
        test("spine_rig_builder imports", False, str(e))
    
    try:
        from veilbreakers_cli import VeilbreakersRigger, NaturalLanguageParser, RigRequest
        test("veilbreakers_cli imports", True)
    except Exception as e:
        test("veilbreakers_cli imports", False, str(e))
    
    # =========================================================================
    # SECTION 2: PART CLASSIFICATION
    # =========================================================================
    section("2. PART CLASSIFICATION")
    
    classifier = PartClassifier()
    
    # Test rigid core parts
    part_type, preset = classifier.classify("head")
    test("head → RIGID_CORE", part_type == PartType.RIGID_CORE)
    
    part_type, preset = classifier.classify("torso")
    test("torso → RIGID_CORE", part_type == PartType.RIGID_CORE)
    
    # Test rigid limbs
    part_type, preset = classifier.classify("arm_left")
    test("arm_left → RIGID_LIMB", part_type == PartType.RIGID_LIMB)
    
    # Test soft parts
    part_type, preset = classifier.classify("hair_long")
    test("hair_long → SOFT_HAIR", part_type == PartType.SOFT_HAIR)
    test("hair_long preset", preset == "hair_long")
    
    part_type, preset = classifier.classify("cape")
    test("cape → SOFT_CLOTH", part_type == PartType.SOFT_CLOTH)
    
    part_type, preset = classifier.classify("tentacle")
    test("tentacle → SOFT_TENTACLE", part_type == PartType.SOFT_TENTACLE)
    
    part_type, preset = classifier.classify("chain")
    test("chain → SOFT_CHAIN", part_type == PartType.SOFT_CHAIN)
    
    part_type, preset = classifier.classify("ghost_orb")
    test("ghost_orb → FLOATING", part_type == PartType.FLOATING)
    
    # =========================================================================
    # SECTION 3: BONE CHAIN GENERATION
    # =========================================================================
    section("3. BONE CHAIN GENERATION")
    
    chain_gen = BoneChainGenerator(classifier)
    
    # Test single bone generation
    bones, ik, physics = chain_gen.generate_chain(
        "head", 0, 100, 50, 0, "root", PartType.RIGID_CORE
    )
    test("single bone count", len(bones) == 1)
    test("single bone name", bones[0].name == "head")
    test("no IK for core", ik is None)
    test("no physics for core", len(physics) == 0)
    
    # Test IK chain generation
    bones, ik, physics = chain_gen.generate_chain(
        "arm", 0, 100, 100, -45, "torso", PartType.RIGID_LIMB, bone_count=3
    )
    test("IK chain bone count", len(bones) == 4)  # 3 bones + IK target
    test("IK constraint created", ik is not None)
    test("IK target exists", any("ik_target" in b.name for b in bones))
    
    # Test physics chain generation
    bones, ik, physics = chain_gen.generate_chain(
        "cape", 0, 0, 100, 180, "torso", PartType.SOFT_CLOTH, 
        preset="cape_light", bone_count=5
    )
    test("physics chain bone count", len(bones) == 5)
    test("physics constraints created", len(physics) > 0)
    test("no IK for physics chain", ik is None)
    
    # =========================================================================
    # SECTION 4: DATA STRUCTURES
    # =========================================================================
    section("4. DATA STRUCTURES")
    
    # Test Bone
    bone = Bone(name="test_bone", parent="root", x=10, y=20, length=50, rotation=45)
    spine_dict = bone.to_spine_dict()
    test("Bone.to_spine_dict()", "name" in spine_dict and spine_dict["name"] == "test_bone")
    test("Bone preserves position", spine_dict.get("x") == 10)
    test("Bone preserves rotation", spine_dict.get("rotation") == 45)
    
    # Test Slot
    slot = Slot(name="test_slot", bone="test_bone", attachment="test_attachment")
    spine_dict = slot.to_spine_dict()
    test("Slot.to_spine_dict()", spine_dict["name"] == "test_slot")
    test("Slot has attachment", spine_dict["attachment"] == "test_attachment")
    
    # Test IKConstraint
    ik = IKConstraint(name="arm_ik", bones=["arm_1", "arm_2"], target="arm_target")
    spine_dict = ik.to_spine_dict()
    test("IKConstraint.to_spine_dict()", spine_dict["name"] == "arm_ik")
    test("IK has bones list", len(spine_dict["bones"]) == 2)
    
    # Test Animation
    from animation_engine import Keyframe, BoneTimeline
    anim = Animation(name="test_anim", duration=1.0)
    anim.bones["root"] = BoneTimeline(bone_name="root")
    anim.bones["root"].rotate = [Keyframe(0.0, 0), Keyframe(1.0, 45)]
    spine_dict = anim.to_spine_dict()
    test("Animation.to_spine_dict()", "bones" in spine_dict)
    test("Animation has bone timeline", "root" in spine_dict["bones"])
    
    # Test CreatureRig
    rig = CreatureRig(name="test_rig", archetype=CreatureArchetype.HUMANOID, width=512, height=512)
    rig.bones.append(Bone(name="root", x=256, y=50))
    rig.slots.append(Slot(name="body", bone="root"))
    json_str = rig.to_spine_json()
    test("CreatureRig.to_spine_json()", len(json_str) > 0)
    
    # Verify JSON is valid
    try:
        parsed = json.loads(json_str)
        test("Spine JSON is valid", True)
        test("JSON has skeleton", "skeleton" in parsed)
        test("JSON has bones", "bones" in parsed)
    except json.JSONDecodeError as e:
        test("Spine JSON is valid", False, str(e))
    
    # =========================================================================
    # SECTION 5: ANIMATION GENERATOR
    # =========================================================================
    section("5. ANIMATION GENERATOR")
    
    # Create a test rig for animation
    test_rig = CreatureRig(name="anim_test", archetype=CreatureArchetype.HUMANOID)
    test_rig.bones = [
        Bone(name="root", x=100, y=50),
        Bone(name="torso", parent="root", x=0, y=50),
        Bone(name="head", parent="torso", x=0, y=40),
        Bone(name="arm_left", parent="torso", x=-30, y=0),
        Bone(name="arm_right", parent="torso", x=30, y=0),
    ]
    
    gen = AnimationGenerator(test_rig)
    
    # Test basic animation creation
    anim = gen.create_animation("test", 2.0)
    test("create_animation()", anim.name == "test" and anim.duration == 2.0)
    
    # Test keyframe generation
    gen.add_rotation_keyframes(anim, "root", [(0.0, 0, "linear"), (1.0, 45, "pow2")])
    test("add_rotation_keyframes()", len(anim.bones["root"].rotate) == 2)
    
    gen.add_translation_keyframes(anim, "torso", [(0.0, 0, 0, "linear"), (1.0, 10, 20, "pow2")])
    test("add_translation_keyframes()", len(anim.bones["torso"].translate) == 2)
    
    gen.add_scale_keyframes(anim, "head", [(0.0, 1.0, 1.0, "linear"), (1.0, 1.2, 1.2, "pow2")])
    test("add_scale_keyframes()", len(anim.bones["head"].scale) == 2)
    
    # Test sine wave generation
    keyframes = gen.generate_sine_wave(2.0, 1.0, 10.0, keyframe_count=10)
    test("generate_sine_wave()", len(keyframes) == 11)
    test("sine wave oscillates", keyframes[0][1] != keyframes[5][1])
    
    # =========================================================================
    # SECTION 6: ANIMATION TEMPLATES
    # =========================================================================
    section("6. ANIMATION TEMPLATES")
    
    templates = AnimationTemplates
    
    # Test idle animations
    anim = templates.idle_breathe(gen, ["torso"], ["head"], 2.0)
    test("idle_breathe template", anim.name == "idle_breathe" and len(anim.bones) > 0)
    
    anim = templates.idle_combat(gen, "root", ["arm_left", "arm_right"], 1.5)
    test("idle_combat template", anim.name == "idle_combat")
    
    anim = templates.idle_float(gen, "root", 2.5)
    test("idle_float template", anim.name == "idle_float")
    
    # Test attack animations
    anim = templates.attack_slash(gen, "root", ["arm_right"], 0.5)
    test("attack_slash template", anim.name == "attack_slash")
    test("attack has hit_frame event", any(e.get("name") == "hit_frame" for e in anim.events))
    
    anim = templates.attack_overhead(gen, "root", ["arm_left", "arm_right"], 0.7)
    test("attack_overhead template", anim.name == "attack_overhead")
    
    # Test death animations
    anim = templates.death_fall_forward(gen, "root", ["body_slot"], 1.0)
    test("death_fall_forward template", anim.name == "death_fall_forward")
    
    anim = templates.death_dissolve(gen, "root", ["body_slot"], 1.5)
    test("death_dissolve template", anim.name == "death_dissolve")
    
    # Test special animations
    anim = templates.special_charge(gen, "root", ["slot1"], 1.5)
    test("special_charge template", anim.name == "special_charge")
    
    anim = templates.spawn(gen, "root", ["slot1"], 1.0)
    test("spawn template", anim.name == "spawn")
    
    # =========================================================================
    # SECTION 7: ARCHETYPE CONFIGURATIONS
    # =========================================================================
    section("7. ARCHETYPE CONFIGURATIONS")
    
    test("ARCHETYPE_CONFIGS exists", len(ARCHETYPE_CONFIGS) > 0)
    
    expected_archetypes = [
        CreatureArchetype.HUMANOID,
        CreatureArchetype.MULTI_ARM,
        CreatureArchetype.QUADRUPED,
        CreatureArchetype.SERPENT,
        CreatureArchetype.SKELETON,
        CreatureArchetype.FLOATING,
        CreatureArchetype.GIANT,
        CreatureArchetype.INSECTOID,
        CreatureArchetype.WINGED,
        CreatureArchetype.ELDRITCH,
    ]
    
    for arch in expected_archetypes:
        exists = arch in ARCHETYPE_CONFIGS
        test(f"{arch.name} config exists", exists)
        if exists:
            config = ARCHETYPE_CONFIGS[arch]
            test(f"{arch.name} has animations", len(config.get("animations", [])) > 0)
    
    # =========================================================================
    # SECTION 8: SPINE RIG BUILDER
    # =========================================================================
    section("8. SPINE RIG BUILDER")
    
    # Create temp directory for output
    temp_dir = tempfile.mkdtemp()
    
    try:
        builder = SpineRigBuilder(output_dir=temp_dir)
        test("SpineRigBuilder instantiation", builder is not None)
        
        # Create a simple test image
        try:
            from PIL import Image
            import numpy as np
            
            # Create 256x256 test image
            img_array = np.zeros((256, 256, 4), dtype=np.uint8)
            img_array[50:200, 80:180] = [128, 128, 128, 255]  # Body
            img_array[20:60, 100:156] = [200, 200, 200, 255]  # Head
            
            test_img_path = os.path.join(temp_dir, "test_monster.png")
            Image.fromarray(img_array).save(test_img_path)
            
            test("Test image created", os.path.exists(test_img_path))
            
            # Build a rig
            output_path = builder.build(
                image_path=test_img_path,
                name="test_monster",
                archetype="humanoid",
                arm_count=2,
                has_hair=True,
                has_cape=True
            )
            
            test("build() returns path", output_path is not None and len(output_path) > 0)
            test("output file exists", os.path.exists(output_path))
            
            # Verify JSON content
            with open(output_path, 'r') as f:
                spine_data = json.load(f)
            
            test("Spine JSON has skeleton", "skeleton" in spine_data)
            test("Spine JSON has bones", len(spine_data.get("bones", [])) > 0)
            test("Spine JSON has slots", len(spine_data.get("slots", [])) > 0)
            test("Spine JSON has animations", len(spine_data.get("animations", {})) > 0)
            
            # Count animations
            anim_count = len(spine_data.get("animations", {}))
            test(f"Has multiple animations ({anim_count})", anim_count >= 10)
            
            # Check for physics bones (hair/cape)
            bone_names = [b["name"] for b in spine_data.get("bones", [])]
            has_hair = any("hair" in name for name in bone_names)
            has_cape = any("cape" in name for name in bone_names)
            test("Has hair bones", has_hair)
            test("Has cape bones", has_cape)
            
        except ImportError:
            warnings.append("PIL not available - skipping image tests")
            test("PIL available", False, "PIL/Pillow not installed")
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # =========================================================================
    # SECTION 9: NATURAL LANGUAGE PARSER
    # =========================================================================
    section("9. NATURAL LANGUAGE PARSER")
    
    parser = NaturalLanguageParser()
    
    # Test archetype detection
    req = parser.parse("Create a skeleton warrior", "test.png")
    test("detects skeleton", req.archetype == "skeleton")
    
    req = parser.parse("Rig this dragon with wings", "test.png")
    test("detects winged/dragon", req.archetype == "winged" or req.has_wings)
    
    req = parser.parse("A floating ghost spirit", "test.png")
    test("detects floating", req.archetype == "floating")
    
    # Test arm count detection
    req = parser.parse("demon with 6 arms", "test.png")
    test("detects 6 arms", req.arm_count == 6)
    
    req = parser.parse("four-armed creature", "test.png")
    test("detects four arms", req.arm_count == 4)
    
    # Test feature detection
    req = parser.parse("warrior with flowing cape and long hair", "test.png")
    test("detects cape", req.has_cape)
    test("detects hair", req.has_hair)
    
    req = parser.parse("beast with a tail", "test.png")
    test("detects tail", req.has_tail)
    
    req = parser.parse("monster with 8 tentacles", "test.png")
    test("detects tentacles", req.tentacle_count == 8)
    
    # Test speed modifiers
    req = parser.parse("slow lumbering giant", "test.png")
    test("detects slow speed", req.animation_speed < 1.0)
    
    req = parser.parse("quick agile assassin", "test.png")
    test("detects fast speed", req.animation_speed > 1.0)
    
    # =========================================================================
    # SECTION 10: FULL INTEGRATION TEST
    # =========================================================================
    section("10. FULL INTEGRATION TEST")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test image
        try:
            from PIL import Image
            import numpy as np
            
            img_array = np.zeros((512, 512, 4), dtype=np.uint8)
            img_array[100:400, 150:350] = [100, 100, 100, 255]
            
            test_img_path = os.path.join(temp_dir, "integration_test.png")
            Image.fromarray(img_array).save(test_img_path)
            
            # Test the full pipeline via CLI interface
            rigger = VeilbreakersRigger(output_dir=temp_dir)
            test("VeilbreakersRigger instantiation", rigger is not None)
            
            # Test with natural language
            outputs = rigger.rig(
                test_img_path,
                description="demon lord with 4 arms and a flowing cape"
            )
            
            test("rig() returns outputs", len(outputs) > 0)
            
            if "spine" in outputs:
                test("Spine output created", os.path.exists(outputs["spine"]))
                
                with open(outputs["spine"], 'r') as f:
                    data = json.load(f)
                
                # Verify the description was parsed correctly
                bone_names = [b["name"] for b in data.get("bones", [])]
                
                # Should have extra arms
                arm_bones = [n for n in bone_names if "arm" in n.lower()]
                test(f"Has multiple arm bones ({len(arm_bones)})", len(arm_bones) >= 4)
                
                # Should have cape
                cape_bones = [n for n in bone_names if "cape" in n.lower()]
                test("Has cape bones", len(cape_bones) > 0)
            
        except ImportError:
            warnings.append("PIL not available - skipping integration test")
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # =========================================================================
    # SECTION 11: PHYSICS PRESETS
    # =========================================================================
    section("11. PHYSICS PRESETS")
    
    expected_presets = [
        "hair_short", "hair_long", "hair_wild",
        "cape_light", "cape_heavy",
        "tentacle_slow", "tentacle_fast",
        "tail_thick", "tail_whip",
        "chain", "flame", "ethereal"
    ]
    
    for preset_name in expected_presets:
        exists = preset_name in PHYSICS_PRESETS
        test(f"preset: {preset_name}", exists)
        if exists:
            preset = PHYSICS_PRESETS[preset_name]
            test(f"  → has bones count", "bones" in preset)
            test(f"  → has gravity", "gravity" in preset)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"\n  ✅ PASSED: {passed}")
    print(f"  ❌ FAILED: {failed}")
    
    if warnings:
        print(f"\n  ⚠️  WARNINGS: {len(warnings)}")
        for w in warnings:
            print(f"     • {w}")
    
    if errors:
        print(f"\n  🔴 ERRORS:")
        for e in errors[:10]:  # Limit to first 10
            print(f"     • {e}")
    
    success_rate = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0
    print(f"\n  Success Rate: {success_rate:.1f}%")
    
    if failed == 0:
        print("\n  🎉 ALL TESTS PASSED! 🎉")
    
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
