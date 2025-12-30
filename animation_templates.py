#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ANIMATION TEMPLATES LIBRARY                               ║
║                                                                              ║
║   100+ Pre-built Animations for All Creature Archetypes                     ║
║                                                                              ║
║   Each archetype includes:                                                   ║
║   • Idle variations (breathing, combat stance, menacing)                    ║
║   • Movement (walk, run, special locomotion)                                ║
║   • Attacks (basic, combo, special, ultimate)                               ║
║   • Reactions (hit, stagger, block)                                         ║
║   • Death sequences (standard, dramatic, dissolve)                          ║
║   • Special abilities (charge, channel, transform)                          ║
║   • Utility (spawn, taunt, victory)                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from animation_engine import (
    Animation, AnimationGenerator, CreatureRig, CreatureArchetype,
    BoneTimeline, Keyframe, PartType
)
from typing import Dict, List, Callable
import math

# =============================================================================
# ANIMATION TEMPLATE DEFINITIONS
# =============================================================================

class AnimationTemplates:
    """Library of animation templates for all creature types"""
    
    # ─────────────────────────────────────────────────────────────────────────
    # IDLE ANIMATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def idle_breathe(gen: AnimationGenerator, body_bones: List[str], 
                     head_bones: List[str], duration: float = 2.0) -> Animation:
        """Gentle breathing idle - universal"""
        anim = gen.create_animation("idle_breathe", duration)
        
        # Body breathing - subtle scale
        for bone in body_bones:
            gen.add_scale_keyframes(anim, bone, [
                (0.0, 1.0, 1.0, "pow2"),
                (duration/2, 1.02, 0.98, "pow2"),
                (duration, 1.0, 1.0, "pow2"),
            ])
        
        # Head subtle bob
        for bone in head_bones:
            gen.add_translation_keyframes(anim, bone, [
                (0.0, 0, 0, "pow2"),
                (duration/2, 0, 3, "pow2"),
                (duration, 0, 0, "pow2"),
            ])
        
        return anim
    
    @staticmethod
    def idle_combat(gen: AnimationGenerator, root_bone: str,
                    arm_bones: List[str], duration: float = 1.5) -> Animation:
        """Combat-ready stance with subtle sway"""
        anim = gen.create_animation("idle_combat", duration)
        
        # Root sway
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2"),
            (duration/4, -2, "pow2"),
            (duration/2, 0, "pow2"),
            (duration*3/4, 2, "pow2"),
            (duration, 0, "pow2"),
        ])
        
        # Arms ready position
        for i, bone in enumerate(arm_bones):
            offset = i * 0.1
            gen.add_rotation_keyframes(anim, bone, [
                (0.0, 5, "pow2"),
                ((duration/2 + offset) % duration, -5, "pow2"),
                (duration, 5, "pow2"),
            ])
        
        return anim
    
    @staticmethod
    def idle_menace(gen: AnimationGenerator, root_bone: str, head_bones: List[str],
                    duration: float = 3.0) -> Animation:
        """Menacing idle with slow movements"""
        anim = gen.create_animation("idle_menace", duration)
        
        # Slow body sway
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2"),
            (duration/2, 3, "pow2"),
            (duration, 0, "pow2"),
        ])
        
        # Head tracking/looking
        for bone in head_bones:
            gen.add_rotation_keyframes(anim, bone, [
                (0.0, 0, "pow2"),
                (duration/3, 10, "pow2"),
                (duration*2/3, -10, "pow2"),
                (duration, 0, "pow2"),
            ])
        
        return anim
    
    @staticmethod
    def idle_float(gen: AnimationGenerator, root_bone: str,
                   duration: float = 2.5) -> Animation:
        """Floating/hovering idle"""
        anim = gen.create_animation("idle_float", duration)
        
        # Hover bob
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2"),
            (duration/2, 0, 15, "pow2"),
            (duration, 0, 0, "pow2"),
        ])
        
        # Gentle rotation
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, -3, "pow2"),
            (duration/2, 3, "pow2"),
            (duration, -3, "pow2"),
        ])
        
        return anim
    
    @staticmethod
    def idle_twitch(gen: AnimationGenerator, all_bones: List[str],
                    duration: float = 2.0) -> Animation:
        """Insectoid/undead twitchy idle"""
        anim = gen.create_animation("idle_twitch", duration)
        
        # Random twitches
        import random
        for bone in all_bones:
            num_twitches = random.randint(1, 3)
            keyframes = [(0.0, 0, "linear")]
            
            for _ in range(num_twitches):
                t = random.uniform(0.2, duration - 0.2)
                angle = random.uniform(-5, 5)
                keyframes.append((t, angle, "stepped"))
                keyframes.append((t + 0.05, 0, "pow2out"))
            
            keyframes.append((duration, 0, "linear"))
            keyframes.sort(key=lambda x: x[0])
            gen.add_rotation_keyframes(anim, bone, keyframes)
        
        return anim

    # ─────────────────────────────────────────────────────────────────────────
    # MOVEMENT ANIMATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def walk_bipedal(gen: AnimationGenerator, root_bone: str,
                     left_leg: List[str], right_leg: List[str],
                     left_arm: List[str], right_arm: List[str],
                     duration: float = 1.0) -> Animation:
        """Standard bipedal walk cycle"""
        anim = gen.create_animation("walk", duration)
        
        # Root bob
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2"),
            (duration/4, 0, 5, "pow2"),
            (duration/2, 0, 0, "pow2"),
            (duration*3/4, 0, 5, "pow2"),
            (duration, 0, 0, "pow2"),
        ])
        
        # Leg swing - left
        if left_leg:
            gen.add_rotation_keyframes(anim, left_leg[0], [
                (0.0, -30, "pow2"),
                (duration/2, 30, "pow2"),
                (duration, -30, "pow2"),
            ])
        
        # Leg swing - right (opposite phase)
        if right_leg:
            gen.add_rotation_keyframes(anim, right_leg[0], [
                (0.0, 30, "pow2"),
                (duration/2, -30, "pow2"),
                (duration, 30, "pow2"),
            ])
        
        # Arm swing - opposite to legs
        if left_arm:
            gen.add_rotation_keyframes(anim, left_arm[0], [
                (0.0, 20, "pow2"),
                (duration/2, -20, "pow2"),
                (duration, 20, "pow2"),
            ])
        
        if right_arm:
            gen.add_rotation_keyframes(anim, right_arm[0], [
                (0.0, -20, "pow2"),
                (duration/2, 20, "pow2"),
                (duration, -20, "pow2"),
            ])
        
        return anim
    
    @staticmethod
    def walk_quadruped(gen: AnimationGenerator, root_bone: str,
                       front_left: List[str], front_right: List[str],
                       back_left: List[str], back_right: List[str],
                       duration: float = 0.8) -> Animation:
        """Four-legged walk cycle"""
        anim = gen.create_animation("walk", duration)
        
        # Root bob and sway
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2"),
            (duration/4, 2, 3, "pow2"),
            (duration/2, 0, 0, "pow2"),
            (duration*3/4, -2, 3, "pow2"),
            (duration, 0, 0, "pow2"),
        ])
        
        # Diagonal gait: FL + BR, then FR + BL
        if front_left:
            gen.add_rotation_keyframes(anim, front_left[0], [
                (0.0, -25, "pow2"), (duration/2, 25, "pow2"), (duration, -25, "pow2"),
            ])
        if back_right:
            gen.add_rotation_keyframes(anim, back_right[0], [
                (0.0, -25, "pow2"), (duration/2, 25, "pow2"), (duration, -25, "pow2"),
            ])
        if front_right:
            gen.add_rotation_keyframes(anim, front_right[0], [
                (0.0, 25, "pow2"), (duration/2, -25, "pow2"), (duration, 25, "pow2"),
            ])
        if back_left:
            gen.add_rotation_keyframes(anim, back_left[0], [
                (0.0, 25, "pow2"), (duration/2, -25, "pow2"), (duration, 25, "pow2"),
            ])
        
        return anim
    
    @staticmethod
    def slither(gen: AnimationGenerator, body_segments: List[str],
                duration: float = 1.5) -> Animation:
        """Serpentine movement"""
        anim = gen.create_animation("slither", duration)
        
        # Wave propagates down body
        for i, bone in enumerate(body_segments):
            phase = i * (math.pi / len(body_segments))
            amplitude = 15 + i * 2  # Increases toward tail
            
            keyframes = gen.generate_sine_wave(duration, 1.0, amplitude, 12, phase)
            gen.add_rotation_keyframes(anim, bone, keyframes)
        
        return anim
    
    @staticmethod
    def run_bipedal(gen: AnimationGenerator, root_bone: str,
                    left_leg: List[str], right_leg: List[str],
                    left_arm: List[str], right_arm: List[str],
                    duration: float = 0.5) -> Animation:
        """Fast running cycle"""
        anim = gen.create_animation("run", duration)
        
        # More pronounced bob
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 5, 0, "pow2in"),
            (duration/4, 0, 10, "pow2out"),
            (duration/2, -5, 0, "pow2in"),
            (duration*3/4, 0, 10, "pow2out"),
            (duration, 5, 0, "pow2in"),
        ])
        
        # Forward lean
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, -10, "pow2"),
            (duration, -10, "pow2"),
        ])
        
        # Exaggerated leg swing
        if left_leg:
            gen.add_rotation_keyframes(anim, left_leg[0], [
                (0.0, -50, "pow2"),
                (duration/2, 50, "pow2"),
                (duration, -50, "pow2"),
            ])
        if right_leg:
            gen.add_rotation_keyframes(anim, right_leg[0], [
                (0.0, 50, "pow2"),
                (duration/2, -50, "pow2"),
                (duration, 50, "pow2"),
            ])
        
        # Arm pump
        if left_arm:
            gen.add_rotation_keyframes(anim, left_arm[0], [
                (0.0, 40, "pow2"),
                (duration/2, -40, "pow2"),
                (duration, 40, "pow2"),
            ])
        if right_arm:
            gen.add_rotation_keyframes(anim, right_arm[0], [
                (0.0, -40, "pow2"),
                (duration/2, 40, "pow2"),
                (duration, -40, "pow2"),
            ])
        
        return anim
    
    @staticmethod
    def shamble(gen: AnimationGenerator, root_bone: str,
                all_limbs: List[str], duration: float = 1.5) -> Animation:
        """Undead shambling walk"""
        anim = gen.create_animation("shamble", duration)
        
        # Uneven, lurching movement
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2"),
            (duration/3, 5, -3, "pow2in"),
            (duration*2/3, -3, 2, "pow2out"),
            (duration, 0, 0, "pow2"),
        ])
        
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, -5, "pow2"),
            (duration/2, 8, "pow2"),
            (duration, -5, "pow2"),
        ])
        
        # Limbs drag and swing unevenly
        import random
        for limb in all_limbs:
            offset = random.uniform(0, 0.3)
            swing = random.uniform(15, 35)
            gen.add_rotation_keyframes(anim, limb, [
                (offset, -swing, "pow2"),
                ((offset + duration/2) % duration, swing, "pow2"),
                (duration, -swing, "pow2"),
            ])
        
        return anim
    
    @staticmethod
    def drift(gen: AnimationGenerator, root_bone: str,
              duration: float = 2.0) -> Animation:
        """Ghostly drifting movement"""
        anim = gen.create_animation("drift", duration)
        
        # Floating, ethereal movement
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2"),
            (duration/4, 10, 8, "pow2"),
            (duration/2, 20, 0, "pow2"),
            (duration*3/4, 10, -8, "pow2"),
            (duration, 0, 0, "pow2"),
        ])
        
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2"),
            (duration/2, 5, "pow2"),
            (duration, 0, "pow2"),
        ])
        
        # Fade in/out slightly
        gen.add_scale_keyframes(anim, root_bone, [
            (0.0, 1.0, 1.0, "pow2"),
            (duration/2, 1.05, 1.05, "pow2"),
            (duration, 1.0, 1.0, "pow2"),
        ])
        
        return anim

    # ─────────────────────────────────────────────────────────────────────────
    # ATTACK ANIMATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def attack_slash(gen: AnimationGenerator, root_bone: str,
                     arm_bones: List[str], duration: float = 0.5) -> Animation:
        """Quick slashing attack"""
        anim = gen.create_animation("attack_slash", duration)
        
        # Windup
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2out"),
            (0.15, 15, "pow2in"),  # Wind up
            (0.25, -10, "pow2out"),  # Slash
            (duration, 0, "pow2"),  # Recovery
        ])
        
        if arm_bones:
            gen.add_rotation_keyframes(anim, arm_bones[0], [
                (0.0, 0, "pow2out"),
                (0.15, -60, "pow2in"),  # Raise arm
                (0.25, 80, "pow2out"),  # Slash down
                (duration, 0, "elastic"),  # Recovery
            ])
        
        # Hit event
        gen.add_event(anim, 0.25, "hit_frame")
        
        return anim
    
    @staticmethod
    def attack_thrust(gen: AnimationGenerator, root_bone: str,
                      arm_bones: List[str], duration: float = 0.4) -> Animation:
        """Thrusting/stabbing attack"""
        anim = gen.create_animation("attack_thrust", duration)
        
        # Body lunge
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2out"),
            (0.1, -20, 0, "pow2in"),  # Pull back
            (0.2, 50, 0, "pow2out"),  # Thrust
            (duration, 0, 0, "elastic"),  # Return
        ])
        
        if arm_bones:
            gen.add_rotation_keyframes(anim, arm_bones[0], [
                (0.0, 0, "pow2"),
                (0.1, -30, "pow2in"),
                (0.2, 20, "pow2out"),
                (duration, 0, "elastic"),
            ])
        
        gen.add_event(anim, 0.2, "hit_frame")
        
        return anim
    
    @staticmethod
    def attack_overhead(gen: AnimationGenerator, root_bone: str,
                        arm_bones: List[str], duration: float = 0.7) -> Animation:
        """Powerful overhead smash"""
        anim = gen.create_animation("attack_overhead", duration)
        
        # Big windup
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2"),
            (0.3, -15, "pow2"),  # Lean back
            (0.5, 20, "pow2in"),  # Slam forward
            (duration, 0, "elastic"),
        ])
        
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2"),
            (0.3, -10, 20, "pow2"),  # Rise up
            (0.5, 20, -10, "pow2in"),  # Slam down
            (duration, 0, 0, "bounce"),
        ])
        
        if arm_bones:
            gen.add_rotation_keyframes(anim, arm_bones[0], [
                (0.0, 0, "pow2"),
                (0.3, -120, "pow2"),  # Arms way back
                (0.5, 90, "pow2in"),  # Slam
                (duration, 0, "elastic"),
            ])
        
        gen.add_event(anim, 0.5, "hit_frame")
        gen.add_event(anim, 0.5, "screen_shake", int_value=2)
        
        return anim
    
    @staticmethod
    def attack_flurry(gen: AnimationGenerator, root_bone: str,
                      arms: List[List[str]], duration: float = 1.2) -> Animation:
        """Multi-arm flurry attack"""
        anim = gen.create_animation("attack_flurry", duration)
        
        # Root oscillates
        keyframes = []
        strikes = len(arms) * 2
        for i in range(strikes + 1):
            t = (i / strikes) * (duration - 0.2)
            angle = 8 if i % 2 == 0 else -8
            keyframes.append((t, angle, "pow2"))
        keyframes.append((duration, 0, "pow2"))
        gen.add_rotation_keyframes(anim, root_bone, keyframes)
        
        # Each arm strikes in sequence
        for i, arm_bones in enumerate(arms):
            if not arm_bones:
                continue
            
            strike_time = (i / len(arms)) * (duration - 0.3)
            
            gen.add_rotation_keyframes(anim, arm_bones[0], [
                (0.0, 0, "pow2"),
                (strike_time, -60, "pow2in"),  # Wind up
                (strike_time + 0.1, 70, "pow2out"),  # Strike
                (strike_time + 0.2, 0, "pow2"),  # Return
                (duration, 0, "pow2"),
            ])
            
            gen.add_event(anim, strike_time + 0.1, "hit_frame", int_value=i)
        
        return anim
    
    @staticmethod
    def attack_bite(gen: AnimationGenerator, root_bone: str,
                    head_bones: List[str], jaw_bone: str,
                    duration: float = 0.5) -> Animation:
        """Biting attack"""
        anim = gen.create_animation("attack_bite", duration)
        
        # Lunge forward
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2"),
            (0.1, -15, 5, "pow2in"),  # Pull back
            (0.25, 40, -5, "pow2out"),  # Lunge
            (duration, 0, 0, "elastic"),
        ])
        
        # Head thrust
        if head_bones:
            gen.add_rotation_keyframes(anim, head_bones[0], [
                (0.0, 0, "pow2"),
                (0.1, 15, "pow2in"),  # Head up
                (0.25, -20, "pow2out"),  # Snap down
                (duration, 0, "pow2"),
            ])
        
        # Jaw snap
        if jaw_bone:
            gen.add_rotation_keyframes(anim, jaw_bone, [
                (0.0, 0, "pow2"),
                (0.15, -30, "pow2in"),  # Open wide
                (0.25, 5, "pow2out"),  # Snap shut
                (duration, 0, "pow2"),
            ])
        
        gen.add_event(anim, 0.25, "hit_frame")
        
        return anim
    
    @staticmethod  
    def attack_pounce(gen: AnimationGenerator, root_bone: str,
                      front_legs: List[str], duration: float = 0.8) -> Animation:
        """Leaping pounce attack"""
        anim = gen.create_animation("attack_pounce", duration)
        
        # Crouch and leap
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2"),
            (0.2, -10, -15, "pow2in"),  # Crouch
            (0.3, 60, 40, "pow2out"),  # Leap
            (0.5, 80, 10, "pow2"),  # Apex
            (0.7, 50, -10, "pow2in"),  # Land
            (duration, 0, 0, "bounce"),
        ])
        
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2"),
            (0.2, 20, "pow2"),  # Crouch angle
            (0.5, -30, "pow2"),  # In air
            (0.7, 10, "pow2"),  # Landing
            (duration, 0, "elastic"),
        ])
        
        # Front legs extend
        for leg in front_legs:
            gen.add_rotation_keyframes(anim, leg, [
                (0.0, 0, "pow2"),
                (0.2, 40, "pow2"),
                (0.5, -60, "pow2"),  # Extended
                (0.7, 20, "pow2"),
                (duration, 0, "pow2"),
            ])
        
        gen.add_event(anim, 0.7, "hit_frame")
        gen.add_event(anim, 0.7, "screen_shake", int_value=1)
        
        return anim
    
    @staticmethod
    def attack_tail_sweep(gen: AnimationGenerator, root_bone: str,
                          tail_bones: List[str], duration: float = 0.6) -> Animation:
        """Sweeping tail attack"""
        anim = gen.create_animation("attack_tail_sweep", duration)
        
        # Body rotates
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2"),
            (0.2, 30, "pow2in"),  # Wind up
            (0.4, -40, "pow2out"),  # Sweep
            (duration, 0, "elastic"),
        ])
        
        # Tail whips through
        for i, bone in enumerate(tail_bones):
            delay = i * 0.03  # Wave effect
            gen.add_rotation_keyframes(anim, bone, [
                (0.0, 0, "pow2"),
                (0.2 + delay, 40 + i * 5, "pow2in"),
                (0.4 + delay, -60 - i * 10, "pow2out"),
                (duration, 0, "elastic"),
            ])
        
        gen.add_event(anim, 0.4, "hit_frame")
        
        return anim
    
    @staticmethod
    def attack_beam(gen: AnimationGenerator, root_bone: str,
                    head_bones: List[str], duration: float = 1.5) -> Animation:
        """Channeled beam attack"""
        anim = gen.create_animation("attack_beam", duration)
        
        # Charge up
        gen.add_scale_keyframes(anim, root_bone, [
            (0.0, 1.0, 1.0, "pow2"),
            (0.5, 1.1, 1.1, "pow2"),  # Swell with power
            (0.6, 1.0, 1.0, "pow2in"),  # Release
            (duration, 1.0, 1.0, "pow2"),
        ])
        
        if head_bones:
            gen.add_rotation_keyframes(anim, head_bones[0], [
                (0.0, 0, "pow2"),
                (0.5, -10, "pow2"),  # Tilt back
                (0.6, 5, "pow2out"),  # Fire
                (duration, 0, "pow2"),
            ])
        
        gen.add_event(anim, 0.3, "charge_start")
        gen.add_event(anim, 0.6, "beam_fire")
        gen.add_event(anim, 1.3, "beam_end")
        
        return anim

    # ─────────────────────────────────────────────────────────────────────────
    # HIT REACTION ANIMATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def hit_light(gen: AnimationGenerator, root_bone: str,
                  all_slots: List[str], duration: float = 0.3) -> Animation:
        """Light hit reaction"""
        anim = gen.create_animation("hit_light", duration)
        
        # Knockback
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2out"),
            (0.1, -15, 0, "pow2out"),
            (duration, 0, 0, "elastic"),
        ])
        
        # Flash white
        for slot in all_slots[:5]:  # Limit to main slots
            gen.add_color_keyframes(anim, slot, [
                (0.0, "FFFFFFFF", "stepped"),
                (0.05, "FFFFFF00", "stepped"),  # Flash
                (0.15, "FFFFFFFF", "pow2"),
            ])
        
        return anim
    
    @staticmethod
    def hit_heavy(gen: AnimationGenerator, root_bone: str,
                  all_slots: List[str], duration: float = 0.5) -> Animation:
        """Heavy hit/stagger"""
        anim = gen.create_animation("hit_heavy", duration)
        
        # Big knockback with rotation
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2out"),
            (0.15, -40, 10, "pow2out"),
            (duration, 0, 0, "elastic"),
        ])
        
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2out"),
            (0.15, 20, "pow2out"),
            (duration, 0, "elastic"),
        ])
        
        # Flash
        for slot in all_slots[:5]:
            gen.add_color_keyframes(anim, slot, [
                (0.0, "FFFFFFFF", "stepped"),
                (0.05, "FF0000FF", "stepped"),  # Red flash
                (0.1, "FFFFFF00", "stepped"),  # White
                (0.2, "FFFFFFFF", "pow2"),
            ])
        
        return anim
    
    @staticmethod
    def hit_launch(gen: AnimationGenerator, root_bone: str,
                   duration: float = 0.8) -> Animation:
        """Launched into air"""
        anim = gen.create_animation("hit_launch", duration)
        
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2out"),
            (0.3, -20, 80, "pow2out"),  # Up
            (0.6, -10, 40, "pow2in"),  # Fall
            (duration, 0, 0, "bounce"),
        ])
        
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2"),
            (0.6, -45, "linear"),
            (duration, 0, "bounce"),
        ])
        
        return anim

    # ─────────────────────────────────────────────────────────────────────────
    # DEATH ANIMATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def death_fall_forward(gen: AnimationGenerator, root_bone: str,
                           all_slots: List[str], duration: float = 1.0) -> Animation:
        """Fall forward death"""
        anim = gen.create_animation("death_fall_forward", duration)
        
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2in"),
            (0.6, 85, "pow2in"),
            (0.7, 80, "pow2out"),  # Slight bounce
            (duration, 90, "pow2"),
        ])
        
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2in"),
            (0.6, 20, -30, "pow2in"),
            (duration, 30, -40, "pow2"),
        ])
        
        # Fade
        for slot in all_slots:
            gen.add_color_keyframes(anim, slot, [
                (0.0, "FFFFFFFF", "pow2"),
                (0.5, "FFFFFFFF", "pow2"),
                (duration, "FFFFFF88", "pow2"),
            ])
        
        gen.add_event(anim, 0.6, "death_impact")
        gen.add_event(anim, duration, "death_complete")
        
        return anim
    
    @staticmethod
    def death_fall_backward(gen: AnimationGenerator, root_bone: str,
                            all_slots: List[str], duration: float = 1.0) -> Animation:
        """Fall backward death"""
        anim = gen.create_animation("death_fall_backward", duration)
        
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2in"),
            (0.5, -70, "pow2in"),
            (0.6, -65, "pow2out"),
            (duration, -75, "pow2"),
        ])
        
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2in"),
            (0.5, -30, -20, "pow2in"),
            (duration, -40, -30, "pow2"),
        ])
        
        for slot in all_slots:
            gen.add_color_keyframes(anim, slot, [
                (0.0, "FFFFFFFF", "pow2"),
                (0.5, "FFFFFFFF", "pow2"),
                (duration, "FFFFFF66", "pow2"),
            ])
        
        gen.add_event(anim, 0.5, "death_impact")
        
        return anim
    
    @staticmethod
    def death_dissolve(gen: AnimationGenerator, root_bone: str,
                       all_slots: List[str], duration: float = 1.5) -> Animation:
        """Dissolve/fade death for ethereal creatures"""
        anim = gen.create_animation("death_dissolve", duration)
        
        # Float up slightly
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2"),
            (duration, 0, 30, "pow2"),
        ])
        
        # Scale down
        gen.add_scale_keyframes(anim, root_bone, [
            (0.0, 1.0, 1.0, "pow2"),
            (duration * 0.8, 1.1, 1.1, "pow2"),
            (duration, 0.0, 0.0, "pow2in"),
        ])
        
        # Fade out
        for slot in all_slots:
            gen.add_color_keyframes(anim, slot, [
                (0.0, "FFFFFFFF", "pow2"),
                (duration * 0.5, "FFFFFF88", "pow2"),
                (duration, "FFFFFF00", "pow2"),
            ])
        
        gen.add_event(anim, duration, "death_complete")
        
        return anim
    
    @staticmethod
    def death_collapse(gen: AnimationGenerator, root_bone: str,
                       limb_bones: List[str], duration: float = 1.2) -> Animation:
        """Skeleton collapse - bones fall apart"""
        anim = gen.create_animation("death_collapse", duration)
        
        # Main body drops
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2in"),
            (0.4, 0, -50, "pow2in"),
            (duration, 0, -60, "pow2"),
        ])
        
        # Limbs scatter
        import random
        for i, bone in enumerate(limb_bones):
            x_dir = random.uniform(-30, 30)
            y_dir = random.uniform(-20, 10)
            rot = random.uniform(-90, 90)
            
            delay = i * 0.05
            gen.add_translation_keyframes(anim, bone, [
                (delay, 0, 0, "pow2out"),
                (0.5 + delay, x_dir, y_dir, "pow2out"),
                (duration, x_dir * 1.2, y_dir - 20, "pow2"),
            ])
            gen.add_rotation_keyframes(anim, bone, [
                (delay, 0, "linear"),
                (duration, rot, "pow2out"),
            ])
        
        gen.add_event(anim, 0.4, "bones_scatter")
        
        return anim
    
    @staticmethod
    def death_explode(gen: AnimationGenerator, root_bone: str,
                      all_slots: List[str], duration: float = 0.5) -> Animation:
        """Explosive death"""
        anim = gen.create_animation("death_explode", duration)
        
        # Quick scale up then disappear
        gen.add_scale_keyframes(anim, root_bone, [
            (0.0, 1.0, 1.0, "pow2out"),
            (0.15, 1.5, 1.5, "pow2out"),
            (0.2, 0.0, 0.0, "stepped"),
        ])
        
        # Flash
        for slot in all_slots:
            gen.add_color_keyframes(anim, slot, [
                (0.0, "FFFFFFFF", "stepped"),
                (0.05, "FFFF00FF", "pow2"),
                (0.15, "FF8800FF", "pow2"),
                (0.2, "FFFFFF00", "stepped"),
            ])
        
        gen.add_event(anim, 0.15, "explosion")
        gen.add_event(anim, 0.15, "screen_shake", int_value=3)
        gen.add_event(anim, duration, "death_complete")
        
        return anim

    # ─────────────────────────────────────────────────────────────────────────
    # SPECIAL ABILITY ANIMATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def special_charge(gen: AnimationGenerator, root_bone: str,
                       all_slots: List[str], duration: float = 1.5) -> Animation:
        """Charging up power"""
        anim = gen.create_animation("special_charge", duration)
        
        # Crouch and gather
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2"),
            (0.5, 0, -10, "pow2"),  # Crouch
            (duration, 0, -10, "pow2"),
        ])
        
        # Pulse scale
        keyframes = []
        pulses = 5
        for i in range(pulses + 1):
            t = (i / pulses) * duration
            scale = 1.0 + (i / pulses) * 0.15  # Growing intensity
            keyframes.append((t, scale, scale, "pow2"))
        gen.add_scale_keyframes(anim, root_bone, keyframes)
        
        # Color shift to power color
        for slot in all_slots[:3]:
            gen.add_color_keyframes(anim, slot, [
                (0.0, "FFFFFFFF", "pow2"),
                (duration * 0.5, "FFFF88FF", "pow2"),
                (duration, "FFCC00FF", "pow2"),
            ])
        
        gen.add_event(anim, 0.0, "charge_start")
        gen.add_event(anim, duration, "charge_ready")
        
        return anim
    
    @staticmethod
    def special_release(gen: AnimationGenerator, root_bone: str,
                        all_slots: List[str], duration: float = 0.8) -> Animation:
        """Release charged power"""
        anim = gen.create_animation("special_release", duration)
        
        # Explosive release
        gen.add_scale_keyframes(anim, root_bone, [
            (0.0, 1.15, 1.15, "pow2out"),
            (0.1, 1.3, 1.3, "pow2out"),
            (0.3, 0.9, 0.9, "elastic"),
            (duration, 1.0, 1.0, "pow2"),
        ])
        
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, -10, "pow2out"),
            (0.1, 0, 20, "pow2out"),
            (duration, 0, 0, "elastic"),
        ])
        
        # Flash
        for slot in all_slots[:3]:
            gen.add_color_keyframes(anim, slot, [
                (0.0, "FFCC00FF", "stepped"),
                (0.1, "FFFFFFFF", "pow2"),
                (0.3, "FFFFFFFF", "pow2"),
                (duration, "FFFFFFFF", "pow2"),
            ])
        
        gen.add_event(anim, 0.1, "special_impact")
        gen.add_event(anim, 0.1, "screen_shake", int_value=2)
        
        return anim
    
    @staticmethod
    def special_transform(gen: AnimationGenerator, root_bone: str,
                          all_slots: List[str], duration: float = 2.0) -> Animation:
        """Transformation sequence"""
        anim = gen.create_animation("special_transform", duration)
        
        # Dramatic scale and rotation
        gen.add_scale_keyframes(anim, root_bone, [
            (0.0, 1.0, 1.0, "pow2"),
            (0.3, 0.8, 1.2, "pow2"),
            (0.5, 1.2, 0.8, "pow2"),
            (1.0, 0.5, 0.5, "pow2in"),  # Shrink
            (1.2, 1.5, 1.5, "pow2out"),  # Burst out
            (duration, 1.0, 1.0, "elastic"),
        ])
        
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2"),
            (1.0, 360, "pow2"),  # Full rotation during shrink
            (duration, 360, "pow2"),
        ])
        
        # Color flash
        for slot in all_slots:
            gen.add_color_keyframes(anim, slot, [
                (0.0, "FFFFFFFF", "pow2"),
                (0.9, "FFFFFF00", "pow2"),  # Fade before burst
                (1.0, "FFFFFFFF", "stepped"),  # Flash
                (1.2, "FFFFFFFF", "pow2"),
            ])
        
        gen.add_event(anim, 1.0, "transform_flash")
        gen.add_event(anim, 1.2, "transform_complete")
        
        return anim
    
    @staticmethod
    def special_roar(gen: AnimationGenerator, root_bone: str,
                     head_bones: List[str], jaw_bone: str,
                     duration: float = 1.2) -> Animation:
        """Roar/scream"""
        anim = gen.create_animation("special_roar", duration)
        
        # Rear back
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2"),
            (0.3, -10, 15, "pow2"),
            (0.5, -5, 10, "pow2"),
            (duration, 0, 0, "pow2"),
        ])
        
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2"),
            (0.3, -15, "pow2"),
            (duration, 0, "pow2"),
        ])
        
        # Head back then forward
        if head_bones:
            gen.add_rotation_keyframes(anim, head_bones[0], [
                (0.0, 0, "pow2"),
                (0.3, -30, "pow2"),
                (0.5, 10, "pow2"),
                (duration, 0, "pow2"),
            ])
        
        # Jaw wide open
        if jaw_bone:
            gen.add_rotation_keyframes(anim, jaw_bone, [
                (0.0, 0, "pow2"),
                (0.3, -40, "pow2"),
                (0.8, -40, "pow2"),
                (duration, 0, "pow2"),
            ])
        
        gen.add_event(anim, 0.4, "roar_sound")
        gen.add_event(anim, 0.4, "screen_shake", int_value=1)
        
        return anim

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITY ANIMATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def spawn(gen: AnimationGenerator, root_bone: str,
              all_slots: List[str], duration: float = 1.0) -> Animation:
        """Spawn/appear animation"""
        anim = gen.create_animation("spawn", duration)
        
        # Scale from nothing
        gen.add_scale_keyframes(anim, root_bone, [
            (0.0, 0.0, 0.0, "pow2out"),
            (0.3, 1.2, 1.2, "pow2out"),
            (0.5, 0.9, 0.9, "pow2"),
            (duration, 1.0, 1.0, "elastic"),
        ])
        
        # Fade in
        for slot in all_slots:
            gen.add_color_keyframes(anim, slot, [
                (0.0, "FFFFFF00", "pow2"),
                (0.3, "FFFFFFFF", "pow2"),
            ])
        
        gen.add_event(anim, 0.0, "spawn_start")
        gen.add_event(anim, duration, "spawn_complete")
        
        return anim
    
    @staticmethod
    def victory(gen: AnimationGenerator, root_bone: str,
                arm_bones: List[str], duration: float = 1.5) -> Animation:
        """Victory pose"""
        anim = gen.create_animation("victory", duration)
        
        # Rise up proud
        gen.add_translation_keyframes(anim, root_bone, [
            (0.0, 0, 0, "pow2out"),
            (0.5, 0, 20, "pow2out"),
            (duration, 0, 15, "pow2"),
        ])
        
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2"),
            (0.5, -5, "pow2"),
            (duration, -5, "pow2"),
        ])
        
        # Arms raised
        for arm in arm_bones:
            gen.add_rotation_keyframes(anim, arm, [
                (0.0, 0, "pow2out"),
                (0.5, -120, "pow2out"),
                (duration, -100, "pow2"),
            ])
        
        gen.add_event(anim, 0.5, "victory_sound")
        
        return anim
    
    @staticmethod
    def taunt(gen: AnimationGenerator, root_bone: str,
              head_bones: List[str], duration: float = 1.2) -> Animation:
        """Taunting gesture"""
        anim = gen.create_animation("taunt", duration)
        
        # Body swagger
        gen.add_rotation_keyframes(anim, root_bone, [
            (0.0, 0, "pow2"),
            (0.3, 10, "pow2"),
            (0.6, -10, "pow2"),
            (0.9, 5, "pow2"),
            (duration, 0, "pow2"),
        ])
        
        # Head nod/shake
        if head_bones:
            gen.add_rotation_keyframes(anim, head_bones[0], [
                (0.0, 0, "pow2"),
                (0.2, -15, "pow2"),
                (0.4, 15, "pow2"),
                (0.6, -10, "pow2"),
                (duration, 0, "pow2"),
            ])
        
        gen.add_event(anim, 0.3, "taunt_sound")
        
        return anim

    # ─────────────────────────────────────────────────────────────────────────
    # PHYSICS/SECONDARY MOTION
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def physics_hair_idle(gen: AnimationGenerator, hair_bones: List[str],
                          duration: float = 2.0) -> Animation:
        """Gentle hair sway for idle"""
        anim = gen.create_animation("physics_hair_idle", duration)
        
        for i, bone in enumerate(hair_bones):
            phase = i * 0.5
            amplitude = 5 + i * 2  # More movement further down
            keyframes = gen.generate_sine_wave(duration, 0.5, amplitude, 12, phase)
            gen.add_rotation_keyframes(anim, bone, keyframes)
        
        return anim
    
    @staticmethod
    def physics_cape_idle(gen: AnimationGenerator, cape_bones: List[str],
                          duration: float = 3.0) -> Animation:
        """Cape flutter in idle"""
        anim = gen.create_animation("physics_cape_idle", duration)
        
        for i, bone in enumerate(cape_bones):
            phase = i * 0.3
            amplitude = 8 + i * 3
            keyframes = gen.generate_sine_wave(duration, 0.4, amplitude, 16, phase)
            gen.add_rotation_keyframes(anim, bone, keyframes)
        
        return anim
    
    @staticmethod
    def physics_tentacle_idle(gen: AnimationGenerator, tentacle_bones: List[str],
                              duration: float = 2.5) -> Animation:
        """Tentacle writhing in idle"""
        anim = gen.create_animation("physics_tentacle_idle", duration)
        
        for i, bone in enumerate(tentacle_bones):
            # Complex wave motion
            phase = i * 0.4
            amplitude = 15 + i * 5
            
            # Primary wave
            keyframes = gen.generate_sine_wave(duration, 0.6, amplitude, 16, phase)
            
            # Add secondary smaller wave
            for j, (t, v, c) in enumerate(keyframes):
                secondary = math.sin(t * 3 + phase * 2) * (amplitude * 0.3)
                keyframes[j] = (t, v + secondary, c)
            
            gen.add_rotation_keyframes(anim, bone, keyframes)
        
        return anim
