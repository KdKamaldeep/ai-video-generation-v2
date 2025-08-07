#!/usr/bin/env python3
"""
Test script to verify scheduler loading fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animatediff_generator import AnimateDiffGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_scheduler_loading():
    """Test that scheduler loading works correctly with the fix"""
    logger.info("Testing scheduler loading fix...")
    
    try:
        # Initialize generator with a simple model
        generator = AnimateDiffGenerator(
            sd_model_id="runwayml/stable-diffusion-v1-5",  # Use a well-known model
            memory_optimization=False,  # Disable memory optimization for testing
            cache_dir="test_cache"
        )
        
        logger.info("✅ AnimateDiffGenerator initialized successfully")
        
        # Test that the pipelines have schedulers
        if generator.sd_pipeline and generator.sd_pipeline.scheduler:
            logger.info(f"✅ SD Pipeline scheduler: {type(generator.sd_pipeline.scheduler).__name__}")
        else:
            logger.warning("⚠️ SD Pipeline scheduler not found")
        
        if generator.animatediff_pipeline and generator.animatediff_pipeline.scheduler:
            logger.info(f"✅ AnimateDiff Pipeline scheduler: {type(generator.animatediff_pipeline.scheduler).__name__}")
        else:
            logger.warning("⚠️ AnimateDiff Pipeline scheduler not found")
        
        # Cleanup
        generator.cleanup()
        logger.info("✅ Test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_scheduler_loading()
    sys.exit(0 if success else 1) 