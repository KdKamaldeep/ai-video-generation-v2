#!/usr/bin/env python3
"""
Image-Only Pipeline with S3 Upload
Fallback option when AnimateDiff is not available
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

def upload_to_s3(file_path, folder="animatediff-images"):
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
    """Run image-only pipeline with S3 upload"""
    logger.info("🖼️  Starting Image-Only Pipeline with S3 Upload")
    
    # Configuration
    text_prompts = [
        "A beautiful sunset over mountains with clouds",
        "A majestic dragon flying over a medieval castle",
        "A serene lake with mountains in the background"
    ]
    
    styles = ["realistic", "fantasy", "cinematic"]
    
    try:
        # Initialize the generator
        logger.info("Initializing AnimateDiffGenerator...")
        generator = AnimateDiffGenerator()
        
        # Check if AnimateDiff is available
        if generator.animatediff_pipeline:
            logger.info("✅ AnimateDiff is available - you can use the full pipeline!")
            logger.info("💡 Try running: python simple_text_to_video.py")
        else:
            logger.info("⚠️  AnimateDiff not available - using image-only mode")
        
        results = []
        
        # Generate images for each prompt
        for i, (prompt, style) in enumerate(zip(text_prompts, styles)):
            logger.info(f"\n🖼️  Generating image {i+1}/{len(text_prompts)}")
            logger.info(f"📝 Prompt: {prompt}")
            logger.info(f"🎨 Style: {style}")
            
            # Generate image
            image_path = generator.generate_image_from_text(
                text=prompt,
                style=style,
                width=512,
                height=768,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=42 + i
            )
            
            if not image_path:
                logger.error(f"❌ Failed to generate image for prompt {i+1}")
                continue
            
            logger.info(f"✅ Image generated: {image_path}")
            
            # Upload to S3
            logger.info("☁️  Uploading to S3...")
            s3_url = upload_to_s3(image_path, "animatediff-images")
            
            results.append({
                "prompt": prompt,
                "style": style,
                "image_path": image_path,
                "s3_url": s3_url
            })
        
        # Success summary
        logger.info("\n" + "="*60)
        logger.info("🎉 IMAGE-ONLY PIPELINE COMPLETED!")
        logger.info("="*60)
        
        logger.info(f"📊 Generated {len(results)} images")
        
        for i, result in enumerate(results):
            logger.info(f"\n🖼️  Image {i+1}:")
            logger.info(f"   📝 Prompt: {result['prompt']}")
            logger.info(f"   🎨 Style: {result['style']}")
            logger.info(f"   📁 Local: {result['image_path']}")
            if result['s3_url']:
                logger.info(f"   ☁️  Cloud: {result['s3_url']}")
            else:
                logger.warning("   ⚠️  S3 upload failed")
        
        # File sizes
        logger.info("\n📊 FILE SIZES:")
        for result in results:
            if os.path.exists(result['image_path']):
                size_mb = os.path.getsize(result['image_path']) / (1024 * 1024)
                logger.info(f"   {os.path.basename(result['image_path'])}: {size_mb:.2f} MB")
        
        # Next steps
        logger.info("\n🚀 NEXT STEPS:")
        logger.info("   • View images in output/images directory")
        logger.info("   • Download from S3 URLs if available")
        logger.info("   • Share S3 URLs for cloud access")
        
        if not generator.animatediff_pipeline:
            logger.info("\n💡 TO ENABLE VIDEO FEATURES:")
            logger.info("   • Install AnimateDiff: pip install diffusers[animatediff]")
            logger.info("   • Run: python simple_text_to_video.py")
        
    except Exception as e:
        logger.error(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 