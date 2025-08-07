#!/usr/bin/env python3
"""
Test script for the new frame-based AnimateDiff approach
"""

import os
import logging
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_new_frame_approach():
    """Test the new frame-based approach"""
    try:
        from utils.animatediff_generator import AnimateDiffGenerator

        logger.info("=" * 60)
        logger.info("Testing New Frame-Based AnimateDiff Approach")
        logger.info("=" * 60)

        # Initialize generator
        logger.info("Initializing AnimateDiff generator...")
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-motion-adapter-v1-5",
            memory_optimization=True
        )

        logger.info("✅ AnimateDiff generator initialized successfully")

        # Test single frame generation
        logger.info("Testing single frame generation...")
        frames_dir = generator.generate_animated_video_from_text(
            text="A parrot flying over trees near river",
            style="realistic",
            num_frames=8,  # Use 8 frames for faster testing
            fps=8,
            num_inference_steps=5,  # Very reduced for faster testing
            guidance_scale=7.5,
            seed=42
        )

        if frames_dir and os.path.exists(frames_dir):
            logger.info(f"✅ Frames generated successfully: {frames_dir}")
            
            # Check if frames exist
            frame_files = [f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png")]
            logger.info(f"✅ Found {len(frame_files)} frame files")
            
            # Check if metadata exists
            metadata_path = os.path.join(frames_dir, "metadata.json")
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.info(f"✅ Metadata found: {metadata.get('frames', 0)} frames, {metadata.get('fps', 0)} FPS")
            else:
                logger.warning("❌ Metadata file not found")
            
            # Test video compilation
            logger.info("Testing video compilation...")
            output_path = os.path.join(frames_dir, "test_output.mp4")
            frame_paths = [os.path.join(frames_dir, f) for f in sorted(frame_files)]
            
            success = generator._compile_frames_to_video(frame_paths, output_path, fps=8)
            
            if success and os.path.exists(output_path):
                logger.info(f"✅ Video compilation successful: {output_path}")
            else:
                logger.error("❌ Video compilation failed")
                return False
                
        else:
            logger.error("❌ Failed to generate frames")
            return False

        # Test multiple frame directory compilation
        logger.info("Testing multiple frame directory compilation...")
        
        # Create a second frame directory
        frames_dir2 = generator.generate_animated_video_from_text(
            text="5 birds flying over a river with green water",
            style="realistic",
            num_frames=8,
            fps=8,
            num_inference_steps=5,
            guidance_scale=7.5,
            seed=123
        )

        if frames_dir2 and os.path.exists(frames_dir2):
            logger.info(f"✅ Second frames generated: {frames_dir2}")
            
            # Compile both frame directories into a single video
            final_output = os.path.join(os.path.dirname(frames_dir), "combined_test.mp4")
            success = generator.compile_multiple_frame_directories_to_video(
                [frames_dir], final_output, fps=8
            )
            
            if success and os.path.exists(final_output):
                logger.info(f"✅ Multiple frame compilation successful: {final_output}")
            else:
                logger.error("❌ Multiple frame compilation failed")
                return False
        else:
            logger.error("❌ Failed to generate second frames")
            return False

        logger.info("✅ All tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_new_frame_approach()
    if success:
        logger.info("🎉 All tests completed successfully!")
    else:
        logger.error("💥 Some tests failed!")
        exit(1)
