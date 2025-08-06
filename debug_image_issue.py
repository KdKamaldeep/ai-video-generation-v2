#!/usr/bin/env python3
"""
Debug script to test image generation and identify the issue
"""

import os
import sys
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_to_s3(file_path: str, file_type: str = "image") -> str:
    """
    Upload generated file to S3
    
    Args:
        file_path: Path to the file to upload
        file_type: Type of file ("image" or "video")
        
    Returns:
        S3 URL of uploaded file or None if failed
    """
    try:
        from utils.s3_uploader import S3Uploader
        
        print(f"📤 Uploading {file_type} to S3...")
        uploader = S3Uploader()
        
        # Determine folder based on file type
        folder = "debug-images" if file_type == "image" else "debug-videos"
        
        # Upload file
        s3_url = uploader.upload_video(
            local_file_path=file_path,
            folder=folder,
            content_type="image/png" 
        )
        
        if s3_url:
            print(f"✅ {file_type.capitalize()} uploaded successfully: {s3_url}")
            return s3_url
        else:
            print(f"❌ Failed to upload {file_type} to S3")
            return None
            
    except Exception as e:
        print(f"❌ Error uploading {file_type} to S3: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_image_generation():
    """Test image generation to see what's happening"""
    print("🔍 Testing Image Generation")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        
        # Initialize generator
        print("1. Initializing Stable Diffusion Generator...")
        generator = StableDiffusionGenerator()
        
        # Test image generation
        print("2. Generating test image...")
        test_prompt = "A person walking in a park on a sunny day"
        
        start_time = time.time()
        image_path = generator.generate_image_from_text(
            text=test_prompt,
            style="realistic"
        )
        generation_time = time.time() - start_time
        
        if image_path:
            print(f"✅ Image generated successfully: {image_path}")
            print(f"   Generation time: {generation_time:.2f} seconds")
            print(f"   File exists: {os.path.exists(image_path)}")
            print(f"   File extension: {os.path.splitext(image_path)[1]}")
            
            # Check file size
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                print(f"   File size: {file_size} bytes")
            
            # Upload to S3
            s3_url = upload_to_s3(image_path, "image")
            if s3_url:
                print(f"   S3 URL: {s3_url}")
            
            return True
        else:
            print("❌ Image generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during image generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_generation():
    """Test video generation from an existing image to debug motion issues"""
    print("\n🎬 Testing Video Generation from Image")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        
        # Initialize generator
        print("1. Initializing Stable Diffusion Generator...")
        generator = StableDiffusionGenerator()
        
        # First generate a test image
        print("2. Generating test image for video...")
        test_prompt = "A person standing in a park on a sunny day"
        
        image_path = generator.generate_image_from_text(
            text=test_prompt,
            style="realistic"
        )
        
        if not image_path or not os.path.exists(image_path):
            print("❌ Failed to generate test image")
            return False
        
        print(f"✅ Test image generated: {image_path}")
        
        # Test video generation with different motion settings
        print("3. Testing video generation with conservative motion...")
        
        start_time = time.time()
        video_path = generator.generate_video_from_image(
            image_path=image_path,
            motion_strength=0.3,  # Very low motion
            num_frames=8,  # Few frames
            fps=8,
            motion_type="subtle",  # Subtle motion
            fast_mode=True
        )
        generation_time = time.time() - start_time
        
        if video_path:
            print(f"✅ Video generated successfully: {video_path}")
            print(f"   Generation time: {generation_time:.2f} seconds")
            print(f"   File exists: {os.path.exists(video_path)}")
            print(f"   File extension: {os.path.splitext(video_path)[1]}")
            
            # Check file size
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                print(f"   File size: {file_size} bytes")
            
            # Upload to S3
            s3_url = upload_to_s3(video_path, "video")
            if s3_url:
                print(f"   S3 URL: {s3_url}")
            
            return True
        else:
            print("❌ Video generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during video generation: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test custom negative prompts
test_image_generation = test_image_generation()
test_video_generation = test_video_generation()
    
print("\n📊 Debug Results:")
print(f"   Image Generation: {'✅' if test_image_generation else '❌'}")
print(f"   Video Generation: {'✅' if test_video_generation else '❌'}")