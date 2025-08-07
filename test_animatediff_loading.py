#!/usr/bin/env python3
"""
Test script to verify AnimateDiff loading works after the fix
"""

import logging
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_animatediff_loading():
    """Test if AnimateDiff can load successfully"""
    
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        logger.info("Testing AnimateDiff loading...")
        
        # Initialize the generator
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-v1-5-2",
            memory_optimization=True
        )
        
        # Check if pipelines loaded successfully
        if generator.sd_pipeline is not None:
            logger.info("✅ Stable Diffusion pipeline loaded successfully")
        else:
            logger.error("❌ Stable Diffusion pipeline failed to load")
            
        if generator.animatediff_pipeline is not None:
            logger.info("✅ AnimateDiff pipeline loaded successfully")
            
            # Test a simple generation
            logger.info("Testing simple video generation...")
            video_path = generator.generate_animated_video_from_text(
                text="A cute cartoon cat",
                style="cartoon",
                width=512,
                height=768,
                num_frames=8,  # Small number for testing
                fps=8,
                motion_strength=0.6,
                num_inference_steps=10,  # Fewer steps for testing
                guidance_scale=7.5,
                seed=42
            )
            
            if video_path:
                logger.info(f"✅ Test video generated successfully: {video_path}")
            else:
                logger.error("❌ Test video generation failed")
        else:
            logger.error("❌ AnimateDiff pipeline failed to load")
            
        # Cleanup
        generator.cleanup()
        logger.info("✅ Cleanup completed")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_animatediff_loading() 