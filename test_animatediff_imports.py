#!/usr/bin/env python3
"""
Test script to check AnimateDiff imports and availability
"""

import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all the imports needed for AnimateDiff"""
    
    logger.info("Testing AnimateDiff imports...")
    
    # Test basic diffusers imports
    try:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        logger.info("✅ StableDiffusionPipeline and DPMSolverMultistepScheduler imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import StableDiffusionPipeline: {e}")
    
    # Test AnimateDiff specific imports
    try:
        from diffusers import AnimateDiffPipeline
        logger.info("✅ AnimateDiffPipeline imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import AnimateDiffPipeline: {e}")
    
    try:
        from diffusers import DDIMScheduler
        logger.info("✅ DDIMScheduler imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import DDIMScheduler: {e}")
    
    try:
        from diffusers.utils import export_to_video
        logger.info("✅ export_to_video imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import export_to_video: {e}")
    
    # Test motion adapter imports
    try:
        from diffusers.models.motion_adapter import MotionAdapter
        logger.info("✅ MotionAdapter imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import MotionAdapter: {e}")
    
    # Check diffusers version
    try:
        import diffusers
        logger.info(f"📦 Diffusers version: {diffusers.__version__}")
    except Exception as e:
        logger.error(f"❌ Could not get diffusers version: {e}")

if __name__ == "__main__":
    test_imports() 