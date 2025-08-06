#!/usr/bin/env python3
"""
Simple Text → Image → Video pipeline using AnimateDiff with S3 upload
Quick test script for the complete workflow including cloud storage
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
    """Run a simple text-to-image-to-video pipeline with S3 upload"""
    logger.info("🎬 Starting Text → Image → Video Pipeline with S3 Upload")
    
    try:
        # Initialize the generator
        logger.info("Initializing AnimateDiffGenerator...")
        generator = AnimateDiffGenerator()
        
        # Your text prompt
        text_prompt = "A beautiful sunset over mountains with clouds moving slowly"
        style = "realistic"
        motion_type = "subtle"
        
        logger.info(f"📝 Text prompt: {text_prompt}")
        logger.info(f"🎨 Style: {style}")
        logger.info(f"🎬 Motion type: {motion_type}")
        
        # Step 1: Generate image from text
        logger.info("\n🖼️  Step 1: Generating image from text...")
        image_path = generator.generate_image_from_text(
            text=text_prompt,
            style=style,
            width=512,
            height=768,
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=42
        )
        
        if not image_path:
            logger.error("❌ Failed to generate image")
            return
        
        logger.info(f"✅ Image generated: {image_path}")
        
        # Step 2: Add motion to create video
        logger.info("\n🎬 Step 2: Adding motion to create video...")
        video_path = generator.add_motion_to_image(
            image_path=image_path,
            motion_type=motion_type,
            num_frames=16,
            fps=8,
            motion_strength=0.8,
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=123
        )
        
        if not video_path:
            logger.error("❌ Failed to generate video")
            return
        
        logger.info(f"✅ Video generated: {video_path}")
        
        # Step 3: Upload to S3
        logger.info("\n☁️  Step 3: Uploading files to S3...")
        
        # Upload image to S3
        image_s3_url = upload_to_s3(image_path, "animatediff-images")
        
        # Upload video to S3
        video_s3_url = upload_to_s3(video_path, "animatediff-videos")
        
        # Success summary
        logger.info("\n" + "="*60)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"📝 Input text: {text_prompt}")
        logger.info(f"🖼️  Generated image: {image_path}")
        logger.info(f"🎬 Generated video: {video_path}")
        logger.info(f"📁 Image file: {os.path.basename(image_path)}")
        logger.info(f"📁 Video file: {os.path.basename(video_path)}")
        
        # S3 URLs
        if image_s3_url:
            logger.info(f"☁️  Image S3 URL: {image_s3_url}")
        else:
            logger.warning("⚠️  Image S3 upload failed or not configured")
            
        if video_s3_url:
            logger.info(f"☁️  Video S3 URL: {video_s3_url}")
        else:
            logger.warning("⚠️  Video S3 upload failed or not configured")
        
        # Check if files exist locally
        if os.path.exists(image_path):
            size_mb = os.path.getsize(image_path) / (1024 * 1024)
            logger.info(f"✅ Image file: {size_mb:.2f} MB")
        else:
            logger.warning("⚠️  Image file not found")
            
        if os.path.exists(video_path):
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            logger.info(f"✅ Video file: {size_mb:.2f} MB")
        else:
            logger.warning("⚠️  Video file not found")
        
        logger.info("\n🚀 Pipeline complete!")
        logger.info(f"   Local Image: {image_path}")
        logger.info(f"   Local Video: {video_path}")
        if image_s3_url:
            logger.info(f"   Cloud Image: {image_s3_url}")
        if video_s3_url:
            logger.info(f"   Cloud Video: {video_s3_url}")
        
    except Exception as e:
        logger.error(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 