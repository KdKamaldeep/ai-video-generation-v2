#!/usr/bin/env python3
"""
Test script for Text → Image → Video pipeline using AnimateDiff
Demonstrates generating an image from text, then adding motion to create a video
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

def test_text_to_image_to_video():
    """Test the complete pipeline: text → image → video"""
    logger.info("=== Testing Text → Image → Video Pipeline ===")
    
    try:
        # Initialize the generator
        generator = AnimateDiffGenerator()
        
        # Test prompts for different scenarios
        test_scenarios = [
            {
                "text": "A serene lake with mountains in the background, golden hour lighting",
                "style": "realistic",
                "motion_type": "subtle",
                "description": "Peaceful landscape with gentle motion"
            },
            {
                "text": "A bustling city street with neon lights and people walking",
                "style": "cinematic", 
                "motion_type": "dynamic",
                "description": "Urban scene with dynamic movement"
            },
            {
                "text": "A butterfly fluttering in a beautiful garden with flowers",
                "style": "artistic",
                "motion_type": "camera_movement",
                "description": "Nature scene with camera motion"
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(test_scenarios):
            logger.info(f"\n--- Scenario {i+1}: {scenario['description']} ---")
            
            # Step 1: Generate image from text
            logger.info(f"Step 1: Generating image from text...")
            logger.info(f"Text: {scenario['text']}")
            
            image_path = generator.generate_image_from_text(
                text=scenario['text'],
                style=scenario['style'],
                width=512,
                height=768,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=42 + i
            )
            
            if not image_path:
                logger.error(f"✗ Failed to generate image for scenario {i+1}")
                continue
            
            logger.info(f"✓ Image generated: {image_path}")
            
            # Step 2: Add motion to create video
            logger.info(f"Step 2: Adding {scenario['motion_type']} motion to create video...")
            
            video_path = generator.add_motion_to_image(
                image_path=image_path,
                motion_type=scenario['motion_type'],
                num_frames=16,
                fps=8,
                motion_strength=0.8,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=123 + i
            )
            
            if video_path:
                logger.info(f"✓ Video generated: {video_path}")
                results.append({
                    "scenario": scenario['description'],
                    "image_path": image_path,
                    "video_path": video_path,
                    "text": scenario['text'],
                    "style": scenario['style'],
                    "motion_type": scenario['motion_type']
                })
            else:
                logger.warning(f"⚠ Failed to generate video for scenario {i+1}")
        
        # Summary
        logger.info(f"\n=== Pipeline Results ===")
        logger.info(f"Total scenarios tested: {len(test_scenarios)}")
        logger.info(f"Successful pipelines: {len(results)}")
        
        for result in results:
            logger.info(f"✓ {result['scenario']}")
            logger.info(f"  Image: {os.path.basename(result['image_path'])}")
            logger.info(f"  Video: {os.path.basename(result['video_path'])}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in text-to-image-to-video test: {e}")
        return []

def test_character_consistency_pipeline():
    """Test the pipeline with character consistency"""
    logger.info("\n=== Testing Character Consistency Pipeline ===")
    
    try:
        # Initialize the generator
        generator = AnimateDiffGenerator()
        
        # Set up character consistency
        character_description = "A young woman with long brown hair, wearing casual clothes, friendly expression"
        generator.set_character_consistency(
            character_description=character_description,
            seed=999
        )
        
        # Test script with consistent character
        script_lines = [
            "She walks through a beautiful garden",
            "She stops to smell the flowers",
            "She smiles as butterflies flutter around her"
        ]
        
        results = []
        
        for i, line in enumerate(script_lines):
            logger.info(f"\n--- Character Scene {i+1} ---")
            logger.info(f"Text: {line}")
            
            # Generate image with character consistency
            image_path = generator.generate_image_from_text(
                text=line,
                style="realistic",
                width=512,
                height=768,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=999 + i  # Use character seed
            )
            
            if not image_path:
                logger.error(f"✗ Failed to generate image for scene {i+1}")
                continue
            
            logger.info(f"✓ Image generated: {image_path}")
            
            # Add motion to create video
            video_path = generator.add_motion_to_image(
                image_path=image_path,
                motion_type="subtle",
                num_frames=16,
                fps=8,
                motion_strength=0.6,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=999 + i
            )
            
            if video_path:
                logger.info(f"✓ Video generated: {video_path}")
                results.append({
                    "scene": f"Scene {i+1}",
                    "text": line,
                    "image_path": image_path,
                    "video_path": video_path
                })
            else:
                logger.warning(f"⚠ Failed to generate video for scene {i+1}")
        
        logger.info(f"\n=== Character Consistency Results ===")
        logger.info(f"Character: {character_description}")
        logger.info(f"Scenes completed: {len(results)}/{len(script_lines)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in character consistency test: {e}")
        return []

def test_direct_video_generation():
    """Test direct video generation from text (for comparison)"""
    logger.info("\n=== Testing Direct Video Generation (for comparison) ===")
    
    try:
        # Initialize the generator
        generator = AnimateDiffGenerator()
        
        # Test direct video generation
        test_prompts = [
            "A butterfly fluttering in a garden",
            "Waves crashing on a beach",
            "Leaves falling from a tree in autumn"
        ]
        
        results = []
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"\n--- Direct Video {i+1} ---")
            logger.info(f"Text: {prompt}")
            
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
                seed=456 + i
            )
            
            if video_path:
                logger.info(f"✓ Direct video generated: {video_path}")
                results.append({
                    "prompt": prompt,
                    "video_path": video_path
                })
            else:
                logger.warning(f"⚠ Failed to generate direct video for prompt {i+1}")
        
        logger.info(f"\n=== Direct Video Results ===")
        logger.info(f"Videos generated: {len(results)}/{len(test_prompts)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in direct video generation test: {e}")
        return []

def main():
    """Run all text-to-image-to-video tests"""
    logger.info("Starting Text → Image → Video pipeline tests...")
    
    # Test 1: Basic text-to-image-to-video pipeline
    pipeline_results = test_text_to_image_to_video()
    
    # Test 2: Character consistency pipeline
    character_results = test_character_consistency_pipeline()
    
    # Test 3: Direct video generation (for comparison)
    direct_results = test_direct_video_generation()
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("=== FINAL SUMMARY ===")
    logger.info(f"✓ Text→Image→Video pipelines: {len(pipeline_results)}")
    logger.info(f"✓ Character consistency scenes: {len(character_results)}")
    logger.info(f"✓ Direct video generations: {len(direct_results)}")
    
    total_generated = len(pipeline_results) + len(character_results) + len(direct_results)
    
    if total_generated > 0:
        logger.info(f"\n🎉 Successfully generated {total_generated} files!")
        logger.info("Check the output/images and output/videos directories for generated files.")
        
        # Show file locations
        logger.info("\n📁 Generated files:")
        for result in pipeline_results:
            logger.info(f"  Image: {result['image_path']}")
            logger.info(f"  Video: {result['video_path']}")
        
        for result in character_results:
            logger.info(f"  Character Image: {result['image_path']}")
            logger.info(f"  Character Video: {result['video_path']}")
        
        for result in direct_results:
            logger.info(f"  Direct Video: {result['video_path']}")
            
    else:
        logger.error("❌ No files were generated. Check the logs for errors.")

if __name__ == "__main__":
    main() 