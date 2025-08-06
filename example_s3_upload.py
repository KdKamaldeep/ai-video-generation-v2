#!/usr/bin/env python3
"""
Example script showing how to upload videos to S3 bucket
"""

import os
import logging
from utils.ffmpeg_video_creator import FFmpegVideoCreator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_existing_video():
    """Upload an existing video file to S3"""
    
    # Path to your video file
    video_path = "output/shorts_with_images_1754407120.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
    
    try:
        # Initialize video creator with S3 upload capability
        creator = FFmpegVideoCreator(use_stable_video_diffusion=False)
        
        # Upload to S3
        logger.info(f"Uploading {video_path} to S3 bucket 'why-would-you/youtube-shorts'")
        s3_url = creator.upload_to_s3(video_path, folder="youtube-shorts")
        
        if s3_url:
            logger.info(f"✅ Success! Video uploaded to: {s3_url}")
            return s3_url
        else:
            logger.error("❌ Upload failed")
            return None
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
    finally:
        if 'creator' in locals():
            creator.cleanup()

def create_and_upload_video():
    """Create a video and upload it to S3 in one step"""
    
    # Example audio and narration data
    audio_path = "output/voice_1754407120.mp3"  # Your audio file
    narration_lines = [
        {"text": "Why would you do that?", "duration": 3.0},
        {"text": "That's just crazy!", "duration": 2.5}
    ]
    
    # Example image paths (if you have them)
    image_paths = [
        "output/images/20250805_130350_80c1ae36_Every_day_is_a_chance_for_a_fr.png",
        "output/images/20250805_151908_6e9bd8c2_POV_shot_moving_towards_a_decr.png"
    ]
    
    try:
        # Initialize video creator
        creator = FFmpegVideoCreator(use_stable_video_diffusion=False)
        
        # Create video and upload to S3 in one step
        logger.info("Creating video and uploading to S3...")
        result = creator.create_and_upload_video(
            audio_path=audio_path,
            narration_lines=narration_lines,
            image_paths=image_paths,
            upload_to_s3=True,
            s3_folder="youtube-shorts"
        )
        
        if result["local_path"] and result["s3_url"]:
            logger.info(f"✅ Success!")
            logger.info(f"📁 Local file: {result['local_path']}")
            logger.info(f"🌐 S3 URL: {result['s3_url']}")
            return result
        else:
            logger.error("❌ Failed to create or upload video")
            return None
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
    finally:
        if 'creator' in locals():
            creator.cleanup()

def main():
    """Main function"""
    logger.info("🚀 S3 Upload Examples")
    logger.info("="*50)
    
    # Example 1: Upload existing video
    logger.info("\n📤 Example 1: Upload existing video")
    logger.info("-" * 30)
    s3_url = upload_existing_video()
    
    if s3_url:
        logger.info(f"Video is now available at: {s3_url}")
    
    # Example 2: Create and upload (commented out as it requires audio/narration data)
    logger.info("\n🎬 Example 2: Create and upload video")
    logger.info("-" * 30)
    logger.info("This example requires audio and narration data.")
    logger.info("Uncomment the line below to test it:")
    logger.info("# result = create_and_upload_video()")
    
    # Uncomment the line below to test create_and_upload_video
    # result = create_and_upload_video()

if __name__ == "__main__":
    main() 