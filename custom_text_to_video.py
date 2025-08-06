#!/usr/bin/env python3
"""
Customizable Text → Image → Video pipeline using AnimateDiff with S3 upload
Modify the variables below to customize your generation
"""

import os
import sys
import logging

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from animatediff_generator import AnimateDiffGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CUSTOMIZE THESE VARIABLES FOR YOUR GENERATION
# ============================================================================

# Text prompt for image generation
TEXT_PROMPT = "A majestic dragon flying over a medieval castle at sunset"

# Image generation settings
IMAGE_STYLE = "fantasy"  # Options: realistic, cinematic, artistic, fantasy, cartoon, minimalist
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 768
IMAGE_STEPS = 20
IMAGE_GUIDANCE = 7.5
IMAGE_SEED = 42

# Video generation settings
MOTION_TYPE = "dynamic"  # Options: subtle, dynamic, camera_movement, object_motion, zoom, pan, tilt, rotation
NUM_FRAMES = 16
FPS = 8
MOTION_STRENGTH = 0.8
VIDEO_STEPS = 20
VIDEO_GUIDANCE = 7.5
VIDEO_SEED = 123

# Character consistency (optional)
USE_CHARACTER_CONSISTENCY = False
CHARACTER_DESCRIPTION = "A brave knight with silver armor and flowing red cape"
CHARACTER_SEED = 999

# S3 upload settings
ENABLE_S3_UPLOAD = True
S3_IMAGE_FOLDER = "animatediff-images"
S3_VIDEO_FOLDER = "animatediff-videos"

# ============================================================================
# END OF CUSTOMIZATION SECTION
# ============================================================================

def upload_to_s3(file_path, folder="animatediff-output"):
    """Upload a file to S3 bucket"""
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
        
        # S3 configuration
        s3_bucket = os.getenv("S3_BUCKET_NAME", "your-bucket-name")
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        
        if not all([aws_access_key, aws_secret_key, s3_bucket]):
            logger.warning("⚠️  S3 credentials not configured. Skipping upload.")
            return None
        
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # Generate S3 key
        filename = os.path.basename(file_path)
        s3_key = f"{folder}/{filename}"
        
        # Upload file
        logger.info(f"📤 Uploading {filename} to S3...")
        s3_client.upload_file(file_path, s3_bucket, s3_key)
        
        # Generate S3 URL
        s3_url = f"https://{s3_bucket}.s3.{aws_region}.amazonaws.com/{s3_key}"
        logger.info(f"✅ Uploaded to S3: {s3_url}")
        
        return s3_url
        
    except NoCredentialsError:
        logger.error("❌ AWS credentials not found. Please configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return None
    except Exception as e:
        logger.error(f"❌ S3 upload failed: {e}")
        return None

def main():
    """Run the customizable text-to-image-to-video pipeline with S3 upload"""
    logger.info("🎬 Starting Customizable Text → Image → Video Pipeline with S3 Upload")
    
    try:
        # Initialize the generator
        logger.info("Initializing AnimateDiffGenerator...")
        generator = AnimateDiffGenerator()
        
        # Set up character consistency if requested
        if USE_CHARACTER_CONSISTENCY:
            logger.info(f"Setting up character consistency: {CHARACTER_DESCRIPTION}")
            generator.set_character_consistency(
                character_description=CHARACTER_DESCRIPTION,
                seed=CHARACTER_SEED
            )
        
        # Display settings
        logger.info(f"📝 Text prompt: {TEXT_PROMPT}")
        logger.info(f"🎨 Image style: {IMAGE_STYLE}")
        logger.info(f"🎬 Motion type: {MOTION_TYPE}")
        logger.info(f"📐 Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
        logger.info(f"🎞️  Video frames: {NUM_FRAMES} at {FPS} FPS")
        logger.info(f"☁️  S3 upload: {'Enabled' if ENABLE_S3_UPLOAD else 'Disabled'}")
        
        # Step 1: Generate image from text
        logger.info("\n🖼️  Step 1: Generating image from text...")
        image_path = generator.generate_image_from_text(
            text=TEXT_PROMPT,
            style=IMAGE_STYLE,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            num_inference_steps=IMAGE_STEPS,
            guidance_scale=IMAGE_GUIDANCE,
            seed=IMAGE_SEED
        )
        
        if not image_path:
            logger.error("❌ Failed to generate image")
            return
        
        logger.info(f"✅ Image generated: {image_path}")
        
        # Step 2: Add motion to create video
        logger.info("\n🎬 Step 2: Adding motion to create video...")
        video_path = generator.add_motion_to_image(
            image_path=image_path,
            motion_type=MOTION_TYPE,
            num_frames=NUM_FRAMES,
            fps=FPS,
            motion_strength=MOTION_STRENGTH,
            num_inference_steps=VIDEO_STEPS,
            guidance_scale=VIDEO_GUIDANCE,
            seed=VIDEO_SEED
        )
        
        if not video_path:
            logger.error("❌ Failed to generate video")
            return
        
        logger.info(f"✅ Video generated: {video_path}")
        
        # Step 3: Upload to S3 (if enabled)
        image_s3_url = None
        video_s3_url = None
        
        if ENABLE_S3_UPLOAD:
            logger.info("\n☁️  Step 3: Uploading files to S3...")
            
            # Upload image to S3
            image_s3_url = upload_to_s3(image_path, S3_IMAGE_FOLDER)
            
            # Upload video to S3
            video_s3_url = upload_to_s3(video_path, S3_VIDEO_FOLDER)
        else:
            logger.info("\n☁️  Step 3: S3 upload disabled - skipping cloud storage")
        
        # Success summary
        logger.info("\n" + "="*70)
        logger.info("🎉 CUSTOM PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"📝 Input text: {TEXT_PROMPT}")
        logger.info(f"🖼️  Generated image: {image_path}")
        logger.info(f"🎬 Generated video: {video_path}")
        logger.info(f"📁 Image file: {os.path.basename(image_path)}")
        logger.info(f"📁 Video file: {os.path.basename(video_path)}")
        
        # File size information
        if os.path.exists(image_path):
            size_mb = os.path.getsize(image_path) / (1024 * 1024)
            logger.info(f"✅ Image file: {size_mb:.2f} MB")
            
        if os.path.exists(video_path):
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            logger.info(f"✅ Video file: {size_mb:.2f} MB")
        
        # S3 URLs
        if image_s3_url:
            logger.info(f"☁️  Image S3 URL: {image_s3_url}")
        else:
            logger.warning("⚠️  Image S3 upload failed or not configured")
            
        if video_s3_url:
            logger.info(f"☁️  Video S3 URL: {video_s3_url}")
        else:
            logger.warning("⚠️  Video S3 upload failed or not configured")
        
        logger.info("\n🚀 Your custom generation is complete!")
        logger.info(f"   Local Image: {image_path}")
        logger.info(f"   Local Video: {video_path}")
        if image_s3_url:
            logger.info(f"   Cloud Image: {image_s3_url}")
        if video_s3_url:
            logger.info(f"   Cloud Video: {video_s3_url}")
        
        # Tips for next time
        logger.info("\n💡 Tips for customization:")
        logger.info("   • Change TEXT_PROMPT for different scenes")
        logger.info("   • Adjust IMAGE_STYLE for different aesthetics")
        logger.info("   • Modify MOTION_TYPE for different movement styles")
        logger.info("   • Increase NUM_FRAMES for longer videos")
        logger.info("   • Adjust MOTION_STRENGTH for more/less movement")
        logger.info("   • Set ENABLE_S3_UPLOAD=False to skip cloud upload")
        logger.info("   • Configure AWS credentials for S3 upload")
        
    except Exception as e:
        logger.error(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 