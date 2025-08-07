#!/usr/bin/env python3
"""
Motion Examples - AnimateDiff

This example demonstrates the different types of motion that AnimateDiff can generate
from text prompts. AnimateDiff uses MotionAdapters to add coherent motion across frames.

Usage: python motion_examples.py
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demonstrate_motion_types():
    """Demonstrate different types of motion that AnimateDiff can generate"""
    logger.info("🎬 AnimateDiff Motion Examples")
    logger.info("=" * 60)
    
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        # Initialize the generator
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-v1-5-2",
            memory_optimization=True
        )
        
        # Different motion types with examples
        motion_examples = [
            {
                "category": "Character Movement",
                "prompts": [
                    "A person walking through a city street, natural movement",
                    "A dancer performing graceful moves, fluid motion",
                    "A cat running through a garden, dynamic movement"
                ]
            },
            {
                "category": "Camera Motion",
                "prompts": [
                    "A camera panning across a beautiful landscape, cinematic movement",
                    "A zoom-in shot of a flower blooming, smooth camera motion",
                    "A tracking shot following a car on a highway, dynamic camera"
                ]
            },
            {
                "category": "Environmental Motion",
                "prompts": [
                    "Waves crashing on a beach, natural water motion",
                    "Leaves falling from trees in autumn, gentle wind motion",
                    "Clouds moving across the sky, atmospheric motion"
                ]
            },
            {
                "category": "Object Animation",
                "prompts": [
                    "A spinning top on a table, rotational motion",
                    "A butterfly flapping its wings, organic motion",
                    "A clock pendulum swinging, mechanical motion"
                ]
            },
            {
                "category": "Abstract Motion",
                "prompts": [
                    "Colorful particles floating in space, abstract motion",
                    "Liquid flowing through transparent tubes, fluid dynamics",
                    "Light rays moving through fog, atmospheric effects"
                ]
            }
        ]
        
        video_paths = []
        
        for category in motion_examples:
            logger.info(f"\n🎯 {category['category']}")
            logger.info("-" * 40)
            
            for i, prompt in enumerate(category['prompts'], 1):
                logger.info(f"  {i}. {prompt}")
                
                # Generate video with motion
                video_path = generator.generate_animated_video_from_text(
                    text=prompt,
                    style="realistic",
                    width=512,
                    height=768,
                    num_frames=16,  # More frames = smoother motion
                    fps=8,
                    motion_strength=0.8,  # Control motion intensity
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    seed=42 + len(video_paths),
                    decode_chunk_size=8
                )
                
                if video_path:
                    video_paths.append({
                        "category": category['category'],
                        "prompt": prompt,
                        "path": video_path
                    })
                    logger.info(f"    ✅ Generated: {os.path.basename(video_path)}")
                else:
                    logger.error(f"    ❌ Failed to generate video")
        
        # Cleanup
        generator.cleanup()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🎉 MOTION GENERATION SUMMARY")
        logger.info("=" * 60)
        
        for category in motion_examples:
            category_videos = [v for v in video_paths if v['category'] == category['category']]
            logger.info(f"{category['category']}: {len(category_videos)} videos generated")
        
        logger.info(f"\nTotal videos generated: {len(video_paths)}")
        logger.info("\n🎬 Motion Types Supported:")
        logger.info("  ✅ Character movement and actions")
        logger.info("  ✅ Camera movements (pan, zoom, track)")
        logger.info("  ✅ Environmental motion (wind, water, clouds)")
        logger.info("  ✅ Object animation (rotation, oscillation)")
        logger.info("  ✅ Abstract motion (particles, fluids, light)")
        
        return video_paths
        
    except Exception as e:
        logger.error(f"❌ Motion examples failed: {e}")
        return []

def demonstrate_motion_parameters():
    """Demonstrate how different parameters affect motion"""
    logger.info("\n🔧 Motion Parameter Examples")
    logger.info("=" * 60)
    
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-v1-5-2",
            memory_optimization=True
        )
        
        base_prompt = "A bird flying through a forest"
        
        # Test different frame counts (affects motion smoothness)
        frame_counts = [8, 16, 24]
        
        for frames in frame_counts:
            logger.info(f"\nTesting {frames} frames:")
            
            video_path = generator.generate_animated_video_from_text(
                text=base_prompt,
                style="realistic",
                num_frames=frames,
                fps=8,
                motion_strength=0.8,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=42,
                decode_chunk_size=8
            )
            
            if video_path:
                logger.info(f"  ✅ {frames} frames: {os.path.basename(video_path)}")
                logger.info(f"     Duration: {frames/8:.1f} seconds")
        
        # Test different FPS (affects playback speed)
        fps_values = [6, 8, 12]
        
        for fps in fps_values:
            logger.info(f"\nTesting {fps} FPS:")
            
            video_path = generator.generate_animated_video_from_text(
                text=base_prompt,
                style="realistic",
                num_frames=16,
                fps=fps,
                motion_strength=0.8,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=42,
                decode_chunk_size=8
            )
            
            if video_path:
                logger.info(f"  ✅ {fps} FPS: {os.path.basename(video_path)}")
                logger.info(f"     Duration: {16/fps:.1f} seconds")
        
        generator.cleanup()
        
        logger.info("\n📊 Parameter Effects:")
        logger.info("  • More frames = smoother, longer motion")
        logger.info("  • Higher FPS = faster playback")
        logger.info("  • Motion strength = intensity of movement")
        logger.info("  • Guidance scale = adherence to prompt")
        
    except Exception as e:
        logger.error(f"❌ Parameter examples failed: {e}")

def explain_motion_technology():
    """Explain how AnimateDiff generates motion"""
    logger.info("\n🔬 How AnimateDiff Motion Works")
    logger.info("=" * 60)
    
    logger.info("🎯 MotionAdapter Technology:")
    logger.info("  • AnimateDiff uses MotionAdapters to add motion to Stable Diffusion")
    logger.info("  • MotionAdapters are trained on video data to understand temporal coherence")
    logger.info("  • They inject motion information into the diffusion process")
    
    logger.info("\n🔄 Motion Generation Process:")
    logger.info("  1. Text prompt is processed by Stable Diffusion")
    logger.info("  2. MotionAdapter adds temporal motion patterns")
    logger.info("  3. Multiple frames are generated with coherent motion")
    logger.info("  4. Frames are combined into a video")
    
    logger.info("\n⚙️ Key Components:")
    logger.info("  • MotionAdapter: Adds motion understanding")
    logger.info("  • DDIMScheduler: Optimized for temporal consistency")
    logger.info("  • Frame generation: Multiple frames with motion")
    logger.info("  • Memory optimization: decode_chunk_size for efficiency")
    
    logger.info("\n🎬 Motion Types Supported:")
    logger.info("  ✅ Natural movement (walking, running, flying)")
    logger.info("  ✅ Camera motion (pan, zoom, track)")
    logger.info("  ✅ Environmental effects (wind, water, fire)")
    logger.info("  ✅ Object animation (rotation, oscillation)")
    logger.info("  ✅ Abstract motion (particles, fluids, light)")

if __name__ == "__main__":
    # Explain the technology
    explain_motion_technology()
    
    # Demonstrate different motion types
    demonstrate_motion_types()
    
    # Show parameter effects
    demonstrate_motion_parameters() 