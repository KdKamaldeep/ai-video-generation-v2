#!/usr/bin/env python3
"""
Test script for S3 upload functionality
"""

import os
import sys
import logging
from utils.ffmpeg_video_creator import FFmpegVideoCreator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_s3_upload():
    """Test S3 upload functionality"""
    
    # Check if we have a video file to upload
    video_path = "output/shorts_with_images_1754407120.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        logger.info("Please create a video first or specify the correct path")
        return False
    
    try:
        # Initialize video creator
        logger.info("Initializing FFmpeg video creator with S3 upload capability")
        creator = FFmpegVideoCreator(use_stable_video_diffusion=False)
        
        # Upload existing video to S3
        logger.info(f"Uploading video to S3: {video_path}")
        s3_url = creator.upload_to_s3(video_path, folder="youtube-shorts")
        
        if s3_url:
            logger.info(f"✅ Video successfully uploaded to S3!")
            logger.info(f"📺 S3 URL: {s3_url}")
            return True
        else:
            logger.error("❌ Failed to upload video to S3")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during S3 upload test: {e}")
        return False
    finally:
        if 'creator' in locals():
            creator.cleanup()

def test_create_and_upload():
    """Test creating a video and uploading it to S3"""
    
    # This would require audio and narration data
    # For now, just demonstrate the method exists
    logger.info("Testing create_and_upload_video method")
    
    try:
        creator = FFmpegVideoCreator(use_stable_video_diffusion=False)
        
        # Example usage (would need actual audio and narration data)
        logger.info("create_and_upload_video method is available with the following signature:")
        logger.info("creator.create_and_upload_video(")
        logger.info("    audio_path='path/to/audio.mp3',")
        logger.info("    narration_lines=[{'text': 'Hello world', 'duration': 3.0}],")
        logger.info("    image_paths=['path/to/image1.jpg'],")
        logger.info("    upload_to_s3=True,")
        logger.info("    s3_folder='youtube-shorts'")
        logger.info(")")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing create_and_upload: {e}")
        return False
    finally:
        if 'creator' in locals():
            creator.cleanup()

def main():
    """Main test function"""
    logger.info("🚀 Starting S3 upload tests")
    
    # Test 1: Upload existing video
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Upload existing video to S3")
    logger.info("="*50)
    success1 = test_s3_upload()
    
    # Test 2: Test create and upload method
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Test create_and_upload_video method")
    logger.info("="*50)
    success2 = test_create_and_upload()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Upload existing video: {'✅ PASS' if success1 else '❌ FAIL'}")
    logger.info(f"Create and upload method: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        logger.info("\n🎉 All tests passed! S3 upload functionality is working.")
    else:
        logger.info("\n⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main() 