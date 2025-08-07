#!/usr/bin/env python3
"""
Kids Cartoon Example - YouTube Shorts

Simple example showing how to generate kid-friendly cartoon content
for YouTube Shorts using our enhanced AnimateDiff stack.

Usage: python kids_cartoon_example.py
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

def simple_kids_cartoon():
    """Generate a simple kids cartoon"""
    logger.info("🎨 Simple Kids Cartoon Example")
    logger.info("=" * 50)
    
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        # Initialize generator
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-v1-5-2",
            memory_optimization=True
        )
        
        # Kid-friendly prompts
        kids_prompts = [
            "A cute cartoon cat playing with colorful balls, child-friendly, safe for children",
            "A friendly cartoon dog running in a sunny garden, bright colors, kid-safe",
            "A magical cartoon butterfly flying around flowers, whimsical, child-friendly"
        ]
        
        video_paths = []
        
        for i, prompt in enumerate(kids_prompts, 1):
            logger.info(f"Generating kids cartoon {i}/{len(kids_prompts)}...")
            
            # Generate cartoon video with kid-appropriate settings
            video_path = generator.generate_animated_video_from_text(
                text=prompt,
                style="cartoon",  # Use cartoon style
                width=512,
                height=768,  # Vertical for YouTube Shorts
                num_frames=16,
                fps=8,
                motion_strength=0.6,  # Gentle motion for kids
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=42 + i,
                decode_chunk_size=8
            )
            
            if video_path:
                video_paths.append(video_path)
                logger.info(f"✅ Kids cartoon {i} generated: {os.path.basename(video_path)}")
            else:
                logger.error(f"❌ Failed to generate kids cartoon {i}")
        
        generator.cleanup()
        
        logger.info("=" * 50)
        logger.info(f"🎉 Generated {len(video_paths)} kids cartoons!")
        logger.info("These are ready for YouTube Shorts!")
        
        return video_paths
        
    except Exception as e:
        logger.error(f"❌ Kids cartoon example failed: {e}")
        return []

def demonstrate_kids_styles():
    """Demonstrate different cartoon styles for kids"""
    logger.info("\n🎨 Kids Cartoon Styles Demo")
    logger.info("=" * 50)
    
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-v1-5-2",
            memory_optimization=True
        )
        
        base_prompt = "A friendly cartoon character saying hello"
        
        # Different cartoon styles for kids
        styles = ["cartoon", "cute", "bright", "simple", "whimsical", "educational"]
        
        for style in styles:
            logger.info(f"Testing {style} style...")
            
            video_path = generator.generate_animated_video_from_text(
                text=base_prompt,
                style=style,
                width=512,
                height=768,
                num_frames=16,
                fps=8,
                motion_strength=0.6,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=42,
                decode_chunk_size=8
            )
            
            if video_path:
                logger.info(f"  ✅ {style} style: {os.path.basename(video_path)}")
            else:
                logger.error(f"  ❌ Failed to generate {style} style")
        
        generator.cleanup()
        
        logger.info("\n📊 Kids Cartoon Styles Available:")
        logger.info("  • cartoon: Classic cartoon style")
        logger.info("  • cute: Adorable and child-friendly")
        logger.info("  • bright: Vibrant and energetic")
        logger.info("  • simple: Clean and minimalist")
        logger.info("  • whimsical: Magical and dreamy")
        logger.info("  • educational: Clear and informative")
        
    except Exception as e:
        logger.error(f"❌ Style demonstration failed: {e}")

def explain_kids_capabilities():
    """Explain what our stack can do for kids content"""
    logger.info("\n🎯 Kids Content Capabilities")
    logger.info("=" * 50)
    
    logger.info("✅ What Our Stack Can Do:")
    logger.info("  🎨 Generate cartoon-style videos")
    logger.info("  🎬 Create animated content with motion")
    logger.info("  📝 Generate kid-friendly scripts")
    logger.info("  🎤 Add voice narration")
    logger.info("  🎵 Combine video and audio")
    logger.info("  ☁️ Upload to cloud storage")
    
    logger.info("\n🎬 YouTube Shorts Features:")
    logger.info("  • Vertical video format (9:16)")
    logger.info("  • 60-second duration")
    logger.info("  • Kid-safe content")
    logger.info("  • Educational themes")
    logger.info("  • Engaging animations")
    
    logger.info("\n🎨 Content Types Supported:")
    logger.info("  • Educational videos")
    logger.info("  • Animal stories")
    logger.info("  • Color and shape learning")
    logger.info("  • Number and counting")
    logger.info("  • Nature exploration")
    logger.info("  • Friendship stories")
    logger.info("  • Creative imagination")
    
    logger.info("\n🔧 Technical Features:")
    logger.info("  • Motion control (gentle for kids)")
    logger.info("  • Style customization")
    logger.info("  • Voice synthesis")
    logger.info("  • Video composition")
    logger.info("  • Cloud storage")
    logger.info("  • Quality optimization")

if __name__ == "__main__":
    # Explain capabilities
    explain_kids_capabilities()
    
    # Generate simple kids cartoon
    simple_kids_cartoon()
    
    # Demonstrate different styles
    demonstrate_kids_styles() 