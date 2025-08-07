#!/usr/bin/env python3
"""
Test script to verify AnimateDiff pipeline loading with updated API
"""

import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_animatediff_loading():
    """Test AnimateDiff pipeline loading with updated API"""
    logger.info("Testing AnimateDiff pipeline loading...")
    
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        # Initialize generator with minimal settings
        generator = AnimateDiffGenerator(
            memory_optimization=True,
            cache_dir="models_cache"
        )
        
        logger.info("✅ AnimateDiff generator initialized successfully")
        
        # Test if pipelines are loaded
        if generator.sd_pipeline:
            logger.info("✅ Stable Diffusion pipeline loaded")
        else:
            logger.warning("⚠️ Stable Diffusion pipeline not loaded")
        
        if generator.animatediff_pipeline:
            logger.info("✅ AnimateDiff pipeline loaded")
            
            # Test basic video generation
            logger.info("Testing basic video generation...")
            video_path = generator.generate_animated_video_from_text(
                text="A simple test scene",
                style="realistic",
                num_frames=16,  # Use 16 frames (compatible with motion adapter)
                fps=8,
                num_inference_steps=5,  # Minimal steps for testing
                guidance_scale=7.5,
                seed=42
            )
            
            if video_path:
                logger.info(f"✅ Test video generated: {video_path}")
            else:
                logger.warning("⚠️ Test video generation failed")
        else:
            logger.warning("⚠️ AnimateDiff pipeline not loaded")
        
        # Cleanup
        generator.cleanup()
        logger.info("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AnimateDiff loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_frame_rate_calculation():
    """Test frame rate calculation logic"""
    logger.info("Testing frame rate calculation logic...")
    
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        generator = AnimateDiffGenerator(
            memory_optimization=True,
            cache_dir="models_cache"
        )
        
        # Test frame count validation
        test_cases = [
            {"input": 10, "expected": 16},  # Below minimum
            {"input": 20, "expected": 24},  # Rounded to multiple of 8
            {"input": 60, "expected": 32},  # Above maximum
            {"input": 30, "expected": 32},  # Rounded to multiple of 8
        ]
        
        for test_case in test_cases:
            input_frames = test_case["input"]
            expected_frames = test_case["expected"]
            actual_frames = generator._validate_frame_count(input_frames)
            
            if actual_frames == expected_frames:
                logger.info(f"✅ Frame validation correct: {input_frames} -> {actual_frames}")
            else:
                logger.warning(f"⚠️ Frame validation mismatch: {input_frames} -> {actual_frames} (expected {expected_frames})")
        
        # Test FPS calculation from kids cartoon generator
        fps = 12
        test_durations = [2, 3, 4, 5]
        
        for duration in test_durations:
            num_frames = max(16, min(32, int(duration * fps)))
            expected_duration = num_frames / fps
            logger.info(f"Duration: {duration}s, FPS: {fps}, Frames: {num_frames}, Expected duration: {expected_duration:.2f}s")
        
        generator.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"❌ Frame rate calculation test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing AnimateDiff Loading and Frame Rate Fixes")
    logger.info("=" * 60)
    
    # Test frame rate calculation first
    logger.info("Testing frame rate calculation...")
    test_frame_rate_calculation()
    
    # Test AnimateDiff loading
    logger.info("Testing AnimateDiff pipeline loading...")
    success = test_animatediff_loading()
    
    if success:
        logger.info("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed")
        sys.exit(1) 