#!/usr/bin/env python3
"""
Direct Text-to-Video Example - AnimateDiff

This example demonstrates how to generate videos directly from text using AnimateDiff,
without needing to create intermediate images first.

Usage: python direct_text_to_video_example.py
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

def direct_text_to_video_example():
    """Example of direct text-to-video generation"""
    logger.info("🎬 Direct Text-to-Video Example")
    logger.info("=" * 50)
    
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        # Initialize the generator
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-v1-5-2",
            memory_optimization=True
        )
        
        # Example prompts for direct video generation
        prompts = [
            "A cat walking through a magical forest, soft lighting",
            "A spaceship flying through colorful nebula, sci-fi style",
            "A butterfly landing on a flower, macro photography style"
        ]
        
        video_paths = []
        
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"Generating video {i}/{len(prompts)}: {prompt[:50]}...")
            
            # Generate video directly from text - no intermediate images needed!
            video_path = generator.generate_animated_video_from_text(
                text=prompt,
                style="realistic",
                width=512,
                height=768,
                num_frames=24,  # Use 24 frames for optimal AnimateDiff compatibility
                fps=8,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=42 + i,
                decode_chunk_size=8  # Memory optimization
            )
            
            if video_path:
                video_paths.append(video_path)
                logger.info(f"✅ Video generated: {os.path.basename(video_path)}")
            else:
                logger.error(f"❌ Failed to generate video for: {prompt}")
        
        # Cleanup
        generator.cleanup()
        
        logger.info("=" * 50)
        logger.info(f"🎉 Generated {len(video_paths)} videos directly from text!")
        logger.info("No intermediate images were needed - AnimateDiff does text-to-video directly!")
        
        return video_paths
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        return []

def compare_approaches():
    """Compare text-to-image vs direct text-to-video approaches"""
    logger.info("🔄 Comparing Approaches")
    logger.info("=" * 50)
    
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-v1-5-2",
            memory_optimization=True
        )
        
        test_prompt = "A beautiful sunset over mountains, cinematic lighting"
        
        logger.info("Approach 1: Text → Image → (manual animation needed)")
        logger.info("  - Generate static image")
        logger.info("  - Would need additional tools for animation")
        
        image_path = generator.generate_image_from_text(
            text=test_prompt,
            style="realistic",
            seed=42
        )
        
        if image_path:
            logger.info(f"  ✅ Static image: {os.path.basename(image_path)}")
        
        logger.info("\nApproach 2: Text → Video (direct)")
        logger.info("  - Generate animated video directly")
        logger.info("  - No intermediate steps needed")
        
        video_path = generator.generate_animated_video_from_text(
            text=test_prompt,
            style="realistic",
            num_frames=24,
            fps=8,
            seed=42,
            decode_chunk_size=8
        )
        
        if video_path:
            logger.info(f"  ✅ Animated video: {os.path.basename(video_path)}")
        
        generator.cleanup()
        
        logger.info("\n🎯 Conclusion:")
        logger.info("  - AnimateDiff can generate videos directly from text")
        logger.info("  - No need for intermediate image generation")
        logger.info("  - More efficient and streamlined workflow")
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")

if __name__ == "__main__":
    # Run the direct text-to-video example
    direct_text_to_video_example()
    
    print("\n" + "="*60 + "\n")
    
    # Compare the two approaches
    compare_approaches() 