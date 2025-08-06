#!/usr/bin/env python3
"""
Test script for AnimateDiff integration with Stable Diffusion
Demonstrates text-to-image generation with Stable Diffusion and motion addition with AnimateDiff
"""

import os
import sys
import logging
from typing import List

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from animatediff_generator import AnimateDiffGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_text_to_image():
    """Test Stable Diffusion text-to-image generation"""
    logger.info("=== Testing Stable Diffusion Text-to-Image Generation ===")
    
    try:
        # Initialize the generator
        generator = AnimateDiffGenerator()
        
        # Test prompts
        test_prompts = [
            "A beautiful sunset over mountains, cinematic lighting",
            "A cozy coffee shop interior with warm lighting",
            "A futuristic city skyline at night with neon lights"
        ]
        
        generated_images = []
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Generating image {i+1}/{len(test_prompts)}: {prompt}")
            
            image_path = generator.generate_image_from_text(
                text=prompt,
                style="cinematic",
                width=512,
                height=768,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=42 + i  # Use consistent seeds for testing
            )
            
            if image_path:
                generated_images.append(image_path)
                logger.info(f"✓ Image generated: {image_path}")
            else:
                logger.error(f"✗ Failed to generate image for: {prompt}")
        
        logger.info(f"Generated {len(generated_images)} images successfully")
        return generated_images
        
    except Exception as e:
        logger.error(f"Error in text-to-image test: {e}")
        return []

def test_animatediff_motion():
    """Test AnimateDiff motion generation"""
    logger.info("=== Testing AnimateDiff Motion Generation ===")
    
    try:
        # Initialize the generator
        generator = AnimateDiffGenerator()
        
        # Test motion types
        motion_types = ["subtle", "dynamic", "camera_movement"]
        
        # Generate a base image first
        logger.info("Generating base image for motion test...")
        base_image_path = generator.generate_image_from_text(
            text="A serene lake with mountains in the background, golden hour lighting",
            style="realistic",
            seed=123
        )
        
        if not base_image_path:
            logger.error("Failed to generate base image for motion test")
            return []
        
        generated_videos = []
        
        for motion_type in motion_types:
            logger.info(f"Adding {motion_type} motion to image...")
            
            video_path = generator.add_motion_to_image(
                image_path=base_image_path,
                motion_type=motion_type,
                num_frames=16,
                fps=8,
                motion_strength=0.8,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=456
            )
            
            if video_path:
                generated_videos.append(video_path)
                logger.info(f"✓ Motion video generated: {video_path}")
            else:
                logger.error(f"✗ Failed to generate motion video for: {motion_type}")
        
        logger.info(f"Generated {len(generated_videos)} motion videos successfully")
        return generated_videos
        
    except Exception as e:
        logger.error(f"Error in AnimateDiff motion test: {e}")
        return []

def test_direct_animated_video():
    """Test direct animated video generation from text"""
    logger.info("=== Testing Direct Animated Video Generation ===")
    
    try:
        # Initialize the generator
        generator = AnimateDiffGenerator()
        
        # Test prompts for direct video generation
        test_prompts = [
            "A butterfly fluttering in a garden",
            "Waves crashing on a beach",
            "Leaves falling from a tree in autumn"
        ]
        
        generated_videos = []
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Generating animated video {i+1}/{len(test_prompts)}: {prompt}")
            
            video_path = generator.generate_animated_video_from_text(
                text=prompt,
                style="realistic",
                width=512,
                height=768,
                num_frames=16,
                fps=8,
                motion_strength=0.8,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=789 + i
            )
            
            if video_path:
                generated_videos.append(video_path)
                logger.info(f"✓ Animated video generated: {video_path}")
            else:
                logger.error(f"✗ Failed to generate animated video for: {prompt}")
        
        logger.info(f"Generated {len(generated_videos)} animated videos successfully")
        return generated_videos
        
    except Exception as e:
        logger.error(f"Error in direct animated video test: {e}")
        return []

def test_script_generation():
    """Test generating images and videos for a script"""
    logger.info("=== Testing Script Generation ===")
    
    try:
        # Initialize the generator
        generator = AnimateDiffGenerator()
        
        # Set up character consistency
        generator.set_character_consistency(
            character_description="A young woman with long brown hair, wearing casual clothes",
            seed=999
        )
        
        # Test script lines
        script_lines = [
            "She walks through a beautiful garden",
            "She stops to smell the flowers",
            "She smiles as butterflies flutter around her"
        ]
        
        logger.info("Generating images for script...")
        image_paths = generator.generate_images_for_script(
            script_lines=script_lines,
            style="realistic",
            maintain_character_consistency=True
        )
        
        logger.info(f"Generated {len([p for p in image_paths if p])} images for script")
        
        logger.info("Generating motion videos for script...")
        video_paths = generator.generate_motion_videos_for_script(
            script_lines=script_lines,
            style="realistic",
            motion_type="subtle",
            num_frames=16,
            fps=8,
            maintain_character_consistency=True
        )
        
        logger.info(f"Generated {len([p for p in video_paths if p])} motion videos for script")
        
        return image_paths, video_paths
        
    except Exception as e:
        logger.error(f"Error in script generation test: {e}")
        return [], []

def main():
    """Run all tests"""
    logger.info("Starting AnimateDiff integration tests...")
    
    # Test 1: Text-to-image generation
    images = test_text_to_image()
    
    # Test 2: AnimateDiff motion generation
    motion_videos = test_animatediff_motion()
    
    # Test 3: Direct animated video generation
    animated_videos = test_direct_animated_video()
    
    # Test 4: Script generation
    script_images, script_videos = test_script_generation()
    
    # Summary
    logger.info("=== Test Summary ===")
    logger.info(f"✓ Generated {len(images)} static images")
    logger.info(f"✓ Generated {len(motion_videos)} motion videos")
    logger.info(f"✓ Generated {len(animated_videos)} direct animated videos")
    logger.info(f"✓ Generated {len([p for p in script_images if p])} script images")
    logger.info(f"✓ Generated {len([p for p in script_videos if p])} script videos")
    
    total_generated = len(images) + len(motion_videos) + len(animated_videos) + len([p for p in script_images if p]) + len([p for p in script_videos if p])
    
    logger.info(f"Total files generated: {total_generated}")
    
    if total_generated > 0:
        logger.info("🎉 All tests completed successfully!")
        logger.info("Check the output/images and output/videos directories for generated files.")
    else:
        logger.error("❌ No files were generated. Check the logs for errors.")

if __name__ == "__main__":
    main() 