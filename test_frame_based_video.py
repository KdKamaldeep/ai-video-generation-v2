#!/usr/bin/env python3
"""
Test script for the new frame-based AnimateDiff video generation
"""

import os
import logging
import tempfile
from PIL import Image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_frame_based_video_generation():
    """Test the new frame-based video generation approach"""
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        logger.info("=" * 60)
        logger.info("Testing Frame-Based AnimateDiff Video Generation")
        logger.info("=" * 60)
        
        # Initialize generator
        logger.info("Initializing AnimateDiff generator...")
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-motion-adapter-v1-5",
            memory_optimization=True
        )
        
        logger.info("✅ AnimateDiff generator initialized successfully")
        
        # Test single frame generation with frame-based approach
        logger.info("Testing single frame generation...")
        frames_dir = generator.generate_animated_video_from_text(
            text="A cute cat sitting in a garden",
            style="cartoon",
            num_frames=16,  # Use 16 frames for faster testing
            fps=8,
            num_inference_steps=10,  # Reduced for faster testing
            guidance_scale=7.5,
            seed=42
        )
        
        if frames_dir and os.path.exists(frames_dir):
            logger.info(f"✅ Single frames generated successfully: {frames_dir}")
            
            # Check if metadata exists in the frames directory
            metadata_path = os.path.join(frames_dir, "metadata.json")
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.info(f"✅ Metadata found in frames directory: {metadata_path}")
            else:
                logger.warning("❌ Metadata file not found in frames directory")
        else:
            logger.error("❌ Failed to generate single frames")
            return False
        
        # Test frame compilation with dummy frames
        logger.info("Testing frame compilation with dummy frames...")
        
        # Create temporary directory with dummy frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create 10 dummy frames
            for i in range(10):
                # Create a simple colored frame
                frame = Image.new('RGB', (512, 768), color=(i * 25, 100, 150))
                frame_path = os.path.join(temp_dir, f"frame_{i+1:04d}.png")
                frame.save(frame_path, "PNG")
            
            # Test compilation
            output_path = os.path.join(temp_dir, "test_output.mp4")
            success = generator._compile_frames_to_video(
                [os.path.join(temp_dir, f"frame_{i+1:04d}.png") for i in range(10)],
                output_path,
                fps=8
            )
            
            if success and os.path.exists(output_path):
                logger.info(f"✅ Frame compilation test successful: {output_path}")
            else:
                logger.error("❌ Frame compilation test failed")
                return False
        
        # Test multiple frame directory compilation
        logger.info("Testing multiple frame directory compilation...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple frame directories
            frame_dirs = []
            for dir_idx in range(3):
                dir_path = os.path.join(temp_dir, f"frames_{dir_idx}")
                os.makedirs(dir_path, exist_ok=True)
                frame_dirs.append(dir_path)
                
                # Create 5 frames in each directory
                for i in range(5):
                    frame = Image.new('RGB', (512, 768), color=(dir_idx * 80, 100, 150))
                    frame_path = os.path.join(dir_path, f"frame_{i+1:04d}.png")
                    frame.save(frame_path, "PNG")
            
            # Test compilation
            output_path = os.path.join(temp_dir, "combined_output.mp4")
            success = generator.compile_multiple_frame_directories_to_video(
                frame_dirs, output_path, fps=8
            )
            
            if success and os.path.exists(output_path):
                logger.info(f"✅ Multiple frame directory compilation successful: {output_path}")
            else:
                logger.error("❌ Multiple frame directory compilation failed")
                return False
        
        logger.info("✅ All frame-based video generation tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_frame_based_video_generation()
    if success:
        logger.info("🎉 All tests completed successfully!")
    else:
        logger.error("💥 Some tests failed!")
        exit(1)
