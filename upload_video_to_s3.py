#!/usr/bin/env python3
"""
Simple script to upload the existing video to S3 bucket
"""

import os
import logging
from utils.ffmpeg_video_creator import FFmpegVideoCreator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Upload the existing video to S3"""
    
    # Path to the existing video
    video_path = "output/shorts_with_images_1754407120.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        logger.info("Please make sure the video file exists before running this script")
        return
    
    try:
        # Initialize video creator with S3 upload capability
        logger.info("Initializing FFmpeg video creator...")
        creator = FFmpegVideoCreator(use_stable_video_diffusion=False)
        
        # Upload to S3
        logger.info(f"Uploading video to S3 bucket 'why-would-you/youtube-shorts'...")
        logger.info(f"Video path: {video_path}")
        
        s3_url = creator.upload_to_s3(
            video_path=video_path,
            folder="youtube-shorts"
        )
        
        if s3_url:
            logger.info("✅ SUCCESS! Video uploaded to S3!")
            logger.info(f"📺 S3 URL: {s3_url}")
            logger.info(f"📁 Local file: {video_path}")
            logger.info(f"🌐 The video is now publicly accessible at the S3 URL above")
        else:
            logger.error("❌ FAILED to upload video to S3")
            logger.error("Please check your AWS credentials and S3 bucket configuration")
            
    except Exception as e:
        logger.error(f"❌ Error during upload: {e}")
        logger.error("Please check your AWS credentials in creds.txt or environment variables")
    finally:
        if 'creator' in locals():
            creator.cleanup()

if __name__ == "__main__":
    logger.info("🚀 Starting S3 video upload...")
    logger.info("=" * 50)
    main()
    logger.info("=" * 50)
    logger.info("Upload process completed!") 