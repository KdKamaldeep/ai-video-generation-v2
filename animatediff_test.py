#!/usr/bin/env python3
"""
Optimized AnimateDiff Test - Direct Text-to-Video Generation

This script demonstrates the complete pipeline using AnimateDiff for direct text-to-video generation:
- Direct text-to-video (no intermediate images needed)
- Memory optimization with decode_chunk_size
- Better parameter handling
- Robust error handling

Based on: https://huggingface.co/docs/diffusers/en/api/pipelines/animatediff

Usage: python animatediff_test.py
"""

import os
import sys
import logging
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_step_1_create_text():
    """Step 1: Create simple text prompts"""
    logger.info("=" * 50)
    logger.info("STEP 1: Creating simple text prompts")
    logger.info("=" * 50)
    
    # Simple test prompts
    prompts = [
        "A cat sitting in a garden, beautiful lighting",
        "A sunset over mountains, cinematic style",
        "A robot in a futuristic city, sci-fi aesthetic"
    ]
    
    logger.info(f"Created {len(prompts)} text prompts:")
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"  {i}. {prompt}")
    
    return prompts

def test_step_2_generate_videos_directly(prompts, generate_images=False):
    """Step 2: Generate videos directly from text using AnimateDiff"""
    logger.info("=" * 50)
    logger.info("STEP 2: Generating videos directly from text")
    logger.info("=" * 50)
    
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        # Initialize enhanced generator
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-v1-5-2",
            memory_optimization=True
        )
        video_paths = []
        image_paths = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}...")
            
            # Optionally generate image for comparison
            if generate_images:
                logger.info(f"  Generating image for comparison...")
                image_path = generator.generate_image_from_text(
                    text=prompt,
                    style="realistic",
                    width=512,
                    height=768,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    seed=42 + i
                )
                if image_path:
                    image_paths.append(image_path)
                    logger.info(f"  ✅ Image generated: {os.path.basename(image_path)}")
            
            # Generate animated video directly from text
            logger.info(f"  Generating video directly from text...")
            video_path = generator.generate_animated_video_from_text(
                text=prompt,
                style="realistic",
                width=512,
                height=768,
                num_frames=24,  # Use 24 frames for optimal AnimateDiff compatibility
                fps=8,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=42 + i,
                decode_chunk_size=8  # Official memory optimization
            )
            
            if video_path:
                video_paths.append(video_path)
                logger.info(f"  ✅ Video generated: {os.path.basename(video_path)}")
            else:
                logger.error(f"  ❌ Failed to generate video {i+1}")
        
        generator.cleanup()
        
        if generate_images:
            logger.info(f"Generated {len(image_paths)} images and {len(video_paths)} videos")
            return video_paths, image_paths
        else:
            logger.info(f"Generated {len(video_paths)} videos directly from text")
            return video_paths
        
    except Exception as e:
        logger.error(f"❌ Error in video generation: {e}")
        return [] if not generate_images else [], []

def test_step_3_combine_with_ffmpeg(video_paths):
    """Step 3: Combine videos using FFmpeg"""
    logger.info("=" * 50)
    logger.info("STEP 3: Combining videos with FFmpeg")
    logger.info("=" * 50)
    
    if not video_paths:
        logger.error("❌ No videos to combine")
        return None
    
    try:
        import ffmpeg
        
        # Create output directory
        output_dir = "output/test_videos"
        os.makedirs(output_dir, exist_ok=True)
        
        # Output path
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"combined_video_{timestamp}.mp4")
        
        logger.info(f"Combining {len(video_paths)} videos...")
        
        # Create a simple concatenation
        # First, create a file list for FFmpeg
        file_list_path = os.path.join(output_dir, "file_list.txt")
        with open(file_list_path, 'w') as f:
            for video_path in video_paths:
                f.write(f"file '{os.path.abspath(video_path)}'\n")
        
        # Use FFmpeg to concatenate videos
        (
            ffmpeg
            .input(file_list_path, f='concat', safe=0)
            .output(output_path, c='copy')
            .overwrite_output()
            .run(quiet=True)
        )
        
        # Clean up file list
        os.remove(file_list_path)
        
        if os.path.exists(output_path):
            logger.info(f"✅ Combined video created: {os.path.basename(output_path)}")
            return output_path
        else:
            logger.error("❌ Failed to create combined video")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error combining videos: {e}")
        return None

def test_step_4_upload_to_s3(video_path):
    """Step 4: Upload video to S3"""
    logger.info("=" * 50)
    logger.info("STEP 4: Uploading to S3")
    logger.info("=" * 50)
    
    if not video_path or not os.path.exists(video_path):
        logger.error(f"❌ Video file not found: {video_path}")
        return None
    
    try:
        from utils.s3_uploader import S3Uploader
        
        # Initialize S3 uploader
        uploader = S3Uploader(bucket_name="why-would-you")
        
        # Generate S3 key
        timestamp = int(time.time())
        filename = os.path.basename(video_path)
        s3_key = f"test-videos/{timestamp}/{filename}"
        
        logger.info(f"Uploading {filename} to S3...")
        
        # Upload to S3
        s3_url = uploader.upload_video(
            local_file_path=video_path,
            s3_key=s3_key,
            folder="test-videos"
        )
        
        if s3_url:
            logger.info(f"✅ Successfully uploaded to S3: {s3_url}")
            return s3_url
        else:
            logger.error("❌ Failed to upload to S3")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error uploading to S3: {e}")
        return None

def main():
    """Main function to run the optimized pipeline test"""
    logger.info("🎬 Optimized AnimateDiff Pipeline Test")
    logger.info("Testing: Text → Direct Video Generation → FFmpeg → S3")
    logger.info("Following official Hugging Face patterns")
    logger.info("=" * 60)
    
    start_time = time.time()
    results = {
        "success": False,
        "steps": {},
        "final_s3_url": None,
        "duration": 0
    }
    
    try:
        # Step 1: Create text
        prompts = test_step_1_create_text()
        results["steps"]["text"] = {"success": True, "count": len(prompts)}
        
        # Step 2: Generate videos directly from text (with optional images)
        # Set generate_images=False for pure text-to-video pipeline
        # Set generate_images=True if you want comparison images
        video_paths = test_step_2_generate_videos_directly(prompts, generate_images=False)
        results["steps"]["videos"] = {"success": len(video_paths) > 0, "count": len(video_paths)}
        
        # Step 3: Combine with FFmpeg
        combined_video = test_step_3_combine_with_ffmpeg(video_paths)
        results["steps"]["ffmpeg"] = {"success": combined_video is not None}
        
        # Step 4: Upload to S3
        s3_url = None
        if combined_video:
            s3_url = test_step_4_upload_to_s3(combined_video)
        
        results["steps"]["s3"] = {"success": s3_url is not None}
        results["final_s3_url"] = s3_url
        
        # Check overall success
        results["success"] = all(step.get("success", False) for step in results["steps"].values())
        results["duration"] = time.time() - start_time
        
        # Print results
        logger.info("=" * 60)
        logger.info("🎉 OPTIMIZED TEST COMPLETED")
        logger.info("=" * 60)
        
        if results["success"]:
            logger.info("✅ All steps completed successfully!")
            logger.info(f"🌐 Final S3 URL: {s3_url}")
        else:
            logger.error("❌ Some steps failed:")
            for step_name, step_result in results["steps"].items():
                if not step_result.get("success", False):
                    logger.error(f"  - {step_name}")
        
        logger.info(f"⏱️  Total duration: {results['duration']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        results["error"] = str(e)
        results["duration"] = time.time() - start_time
    
    return results

if __name__ == "__main__":
    main() 