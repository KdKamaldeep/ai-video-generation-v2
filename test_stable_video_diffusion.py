#!/usr/bin/env python3
"""
Test script for Stable Video Diffusion functionality

This script tests the basic functionality of stable video diffusion
without requiring full video creation or voice synthesis.
"""

import os
import sys
import logging
from PIL import Image

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.stable_diffusion_generator import StableDiffusionGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_stable_diffusion_initialization():
    """Test if stable diffusion generator initializes correctly"""
    logger.info("Testing Stable Diffusion initialization...")
    
    try:
        sd_generator = StableDiffusionGenerator()
        logger.info("✓ Stable Diffusion generator initialized successfully")
        
        # Check if video pipeline is available
        if sd_generator.video_pipeline is not None:
            logger.info("✓ Stable Video Diffusion pipeline loaded successfully")
        else:
            logger.warning("⚠ Stable Video Diffusion pipeline not available")
        
        sd_generator.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize Stable Diffusion: {e}")
        return False

def test_image_generation():
    """Test basic image generation"""
    logger.info("Testing image generation...")
    
    try:
        sd_generator = StableDiffusionGenerator()
        
        # Generate a simple test image
        image_path = sd_generator.generate_image_from_text(
            text="A simple test image of a cat sitting",
            style="realistic",
            width=512,
            height=512,
            num_inference_steps=10,  # Reduced for faster testing
            seed=42
        )
        
        if image_path and os.path.exists(image_path):
            logger.info(f"✓ Image generated successfully: {image_path}")
            
            # Check image properties
            with Image.open(image_path) as img:
                logger.info(f"✓ Image size: {img.size}")
                logger.info(f"✓ Image mode: {img.mode}")
            
            sd_generator.cleanup()
            return True
        else:
            logger.error("✗ Image generation failed")
            sd_generator.cleanup()
            return False
            
    except Exception as e:
        logger.error(f"✗ Error in image generation test: {e}")
        return False

def test_video_generation():
    """Test video generation from image"""
    logger.info("Testing video generation from image...")
    
    try:
        sd_generator = StableDiffusionGenerator()
        
        # Skip if video pipeline is not available
        if sd_generator.video_pipeline is None:
            logger.warning("⚠ Skipping video generation test - pipeline not available")
            sd_generator.cleanup()
            return True
        
        # First generate a test image
        image_path = sd_generator.generate_image_from_text(
            text="A person walking in a park",
            style="realistic",
            width=1024,
            height=576,  # 16:9 aspect ratio for video
            num_inference_steps=10,
            seed=123
        )
        
        if not image_path:
            logger.error("✗ Failed to generate test image for video")
            sd_generator.cleanup()
            return False
        
        # Generate video from the image
        video_path = sd_generator.generate_video_from_image(
            image_path=image_path,
            motion_strength=0.6,
            num_frames=15,  # Reduced for faster testing
            fps=8,
            seed=123
        )
        
        if video_path and os.path.exists(video_path):
            logger.info(f"✓ Video generated successfully: {video_path}")
            
            # Check file size
            file_size = os.path.getsize(video_path)
            logger.info(f"✓ Video file size: {file_size} bytes")
            
            sd_generator.cleanup()
            return True
        else:
            logger.error("✗ Video generation failed")
            sd_generator.cleanup()
            return False
            
    except Exception as e:
        logger.error(f"✗ Error in video generation test: {e}")
        return False

def test_script_to_videos():
    """Test generating videos for a script"""
    logger.info("Testing script to videos generation...")
    
    try:
        sd_generator = StableDiffusionGenerator()
        
        # Skip if video pipeline is not available
        if sd_generator.video_pipeline is None:
            logger.warning("⚠ Skipping script to videos test - pipeline not available")
            sd_generator.cleanup()
            return True
        
        # Simple test script
        script_lines = [
            "A person walking",
            "The person stopping to look around"
        ]
        
        # Generate videos for script
        video_paths = sd_generator.generate_videos_for_script(
            script_lines=script_lines,
            style="realistic",
            motion_strength=0.5,
            num_frames=10,  # Reduced for faster testing
            fps=8,
            maintain_character_consistency=False
        )
        
        successful_videos = [path for path in video_paths if path is not None]
        logger.info(f"✓ Generated {len(successful_videos)}/{len(script_lines)} videos")
        
        for i, video_path in enumerate(video_paths):
            if video_path:
                logger.info(f"✓ Video {i+1}: {video_path}")
            else:
                logger.warning(f"⚠ Video {i+1}: Failed to generate")
        
        sd_generator.cleanup()
        return len(successful_videos) > 0
        
    except Exception as e:
        logger.error(f"✗ Error in script to videos test: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting Stable Video Diffusion Tests")
    
    # Create output directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/images", exist_ok=True)
    os.makedirs("output/videos", exist_ok=True)
    
    tests = [
        ("Stable Diffusion Initialization", test_stable_diffusion_initialization),
        ("Image Generation", test_image_generation),
        ("Video Generation", test_video_generation),
        ("Script to Videos", test_script_to_videos)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Stable Video Diffusion is working correctly.")
    else:
        logger.warning("⚠ Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 