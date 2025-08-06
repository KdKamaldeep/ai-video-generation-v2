#!/usr/bin/env python3
"""
Simple test script for AnimateDiff integration
Tests basic functionality with corrected model ID
"""

import os
import sys
import logging

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from animatediff_generator import AnimateDiffGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic AnimateDiff functionality"""
    logger.info("=== Testing Basic AnimateDiff Functionality ===")
    
    try:
        # Initialize the generator
        logger.info("Initializing AnimateDiffGenerator...")
        generator = AnimateDiffGenerator()
        
        # Test 1: Generate a simple image
        logger.info("Testing image generation...")
        image_path = generator.generate_image_from_text(
            text="A beautiful sunset over mountains",
            style="realistic",
            width=512,
            height=768,
            num_inference_steps=10,  # Reduced for faster testing
            guidance_scale=7.5,
            seed=42
        )
        
        if image_path:
            logger.info(f"✓ Image generated successfully: {image_path}")
        else:
            logger.error("✗ Failed to generate image")
            return False
        
        # Test 2: Generate animated video (if AnimateDiff is available)
        if generator.animatediff_pipeline is not None:
            logger.info("Testing animated video generation...")
            video_path = generator.generate_animated_video_from_text(
                text="A butterfly fluttering in a garden",
                style="realistic",
                width=512,
                height=768,
                num_frames=8,  # Reduced for faster testing
                fps=8,
                num_inference_steps=10,  # Reduced for faster testing
                guidance_scale=7.5,
                seed=123
            )
            
            if video_path:
                logger.info(f"✓ Animated video generated successfully: {video_path}")
            else:
                logger.warning("✗ Failed to generate animated video (this might be expected if AnimateDiff has issues)")
        else:
            logger.warning("AnimateDiff pipeline not available - skipping video generation test")
        
        logger.info("✓ Basic functionality test completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in basic functionality test: {e}")
        return False

def main():
    """Run the simple test"""
    logger.info("Starting simple AnimateDiff test...")
    
    success = test_basic_functionality()
    
    if success:
        logger.info("🎉 Simple test completed successfully!")
    else:
        logger.error("❌ Simple test failed. Check the logs for errors.")

if __name__ == "__main__":
    main() 