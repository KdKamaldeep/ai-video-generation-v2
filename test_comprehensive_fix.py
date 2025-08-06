#!/usr/bin/env python3
"""
Comprehensive test script to verify image and motion generation fixes with S3 upload
"""

import os
import sys
import time
import logging
import cv2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_to_s3(file_path: str, s3_key: str = None) -> str:
    """Upload a file to S3 and return the S3 URL"""
    try:
        from utils.s3_uploader import S3Uploader
        
        # Initialize S3 uploader
        s3_uploader = S3Uploader()
        
        # Generate S3 key if not provided
        if s3_key is None:
            filename = os.path.basename(file_path)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            s3_key = f"test_results/{timestamp}_{filename}"
        
        # Upload file using the correct method name
        s3_url = s3_uploader.upload_video(file_path, s3_key, folder="test_results")
        
        if s3_url:
            print(f"✅ Uploaded to S3: {s3_url}")
            return s3_url
        else:
            print("❌ S3 upload failed")
            return None
            
    except Exception as e:
        print(f"❌ S3 upload error: {e}")
        return None

def test_image_generation():
    """Test image generation with simplified prompts and S3 upload"""
    print("🖼️ Testing Image Generation")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        
        # Initialize generator
        print("1. Initializing StableDiffusionGenerator...")
        generator = StableDiffusionGenerator()
        
        if generator.pipeline is None:
            print("❌ Image pipeline not available")
            return False
        
        print("✅ Image pipeline loaded")
        
        # Test image generation
        print("2. Testing image generation...")
        test_prompt = "A person walking in a park on a sunny day"
        
        start_time = time.time()
        image_path = generator.generate_image_from_text(
            text=test_prompt,
            style="realistic",
            width=512,
            height=768
        )
        end_time = time.time()
        generation_time = end_time - start_time
        
        if image_path and os.path.exists(image_path):
            print(f"✅ Image generated successfully: {image_path}")
            print(f"   Generation time: {generation_time:.2f} seconds")
            print(f"   File size: {os.path.getsize(image_path) / 1024:.1f} KB")
            
            # Check image properties
            img = cv2.imread(image_path)
            if img is not None:
                height, width, channels = img.shape
                print(f"   Image dimensions: {width}x{height}")
                print(f"   Channels: {channels}")
                
                # Upload to S3
                print("3. Uploading image to S3...")
                s3_url = upload_to_s3(image_path, f"test_results/image_generation_{int(time.time())}.png")
                
                return True
            else:
                print("❌ Could not read generated image")
                return False
        else:
            print("❌ Image generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_motion_generation():
    """Test motion generation with enhanced settings and S3 upload"""
    print("\n🎬 Testing Motion Generation")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        
        # Initialize generator
        print("1. Initializing StableDiffusionGenerator...")
        generator = StableDiffusionGenerator()
        
        if generator.video_pipeline is None:
            print("❌ Video pipeline not available")
            return False
        
        print("✅ Video pipeline loaded")
        
        # Find or create a test image
        test_image_path = None
        if os.path.exists("output/images"):
            image_files = [f for f in os.listdir("output/images") if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                test_image_path = os.path.join("output/images", image_files[0])
                print(f"2. Using existing image: {test_image_path}")
        
        if test_image_path is None:
            print("❌ No test image found")
            return False
        
        # Test motion video generation
        print("3. Testing motion video generation...")
        start_time = time.time()
        
        video_path = generator.generate_video_from_image(
            image_path=test_image_path,
            motion_strength=1.0,
            num_frames=12,
            fps=8,
            motion_type="dynamic",
            fast_mode=True
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        if video_path and os.path.exists(video_path):
            print(f"✅ Motion video generated: {video_path}")
            print(f"   Generation time: {generation_time:.2f} seconds")
            print(f"   File size: {os.path.getsize(video_path) / 1024:.1f} KB")
            
            # Analyze the video
            print("4. Analyzing generated video...")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            print(f"   Video FPS: {fps}")
            print(f"   Frame count: {frame_count}")
            print(f"   Duration: {duration:.2f} seconds")
            
            # Check for motion by comparing frames
            motion_detected = False
            motion_score = 0.0
            
            if frame_count > 1:
                cap = cv2.VideoCapture(video_path)
                ret, first_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                ret, last_frame = cap.read()
                cap.release()
                
                if ret and first_frame is not None and last_frame is not None:
                    # Convert to grayscale for comparison
                    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                    last_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate difference
                    diff = cv2.absdiff(first_gray, last_gray)
                    motion_score = np.mean(diff)
                    
                    print(f"   Motion score: {motion_score:.2f}")
                    motion_detected = motion_score > 5.0
                    print(f"   Motion detected: {'Yes' if motion_detected else 'No'}")
                    
                    if motion_detected:
                        print("✅ Good motion detected!")
                    else:
                        print("⚠️  Motion score too low - may need adjustment")
                else:
                    print("❌ Could not read video frames")
                    return False
            else:
                print("❌ Only one frame in video - no motion possible")
                return False
            
            # Upload to S3
            print("5. Uploading video to S3...")
            s3_url = upload_to_s3(video_path, f"test_results/motion_generation_{int(time.time())}.mp4")
            
            return motion_detected
            
        else:
            print("❌ Motion video generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ffmpeg_integration():
    """Test FFmpeg integration with motion generation and S3 upload"""
    print("\n🎥 Testing FFmpeg Integration")
    print("=" * 50)
    
    try:
        from utils.ffmpeg_video_creator import FFmpegVideoCreator
        
        # Initialize FFmpeg creator
        print("1. Initializing FFmpegVideoCreator...")
        ffmpeg_creator = FFmpegVideoCreator(use_stable_video_diffusion=True)
        
        if ffmpeg_creator.sd_generator is None:
            print("❌ Stable Diffusion generator not available")
            return False
        
        print("✅ FFmpegVideoCreator initialized")
        
        # Test motion video segment creation
        print("2. Testing motion video segment creation...")
        
        # Create a test line
        test_line = {
            "text": "A person walking in a park",
            "duration": 3.0,
            "visual_suggestion": "A person walking in a park on a sunny day"
        }
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = ffmpeg_creator._create_motion_video_segment(
                temp_dir=temp_dir,
                line=test_line,
                index=0,
                motion_strength=1.0,
                num_frames=12,
                fps=8
            )
            
            if video_path and os.path.exists(video_path):
                print(f"✅ Motion video segment created: {video_path}")
                
                # Check if it's a video file
                if video_path.endswith('.mp4'):
                    print("✅ Generated file is a video (MP4)")
                    
                    # Analyze the video
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    
                    print(f"   Video FPS: {fps}")
                    print(f"   Frame count: {frame_count}")
                    print(f"   Duration: {duration:.2f} seconds")
                    
                    # Upload to S3
                    print("3. Uploading FFmpeg video to S3...")
                    s3_url = upload_to_s3(video_path, f"test_results/ffmpeg_integration_{int(time.time())}.mp4")
                    
                    return True
                else:
                    print("❌ Generated file is not a video")
                    return False
            else:
                print("❌ Motion video segment creation failed")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_pipeline():
    """Test the full pipeline with S3 upload"""
    print("\n🚀 Testing Full Pipeline")
    print("=" * 50)
    
    try:
        from main import full_pipeline_with_images
        from main import FullPipelineWithImagesRequest
        
        print("1. Running full pipeline...")
        start_time = time.time()
        
        # Create the request object with correct parameters
        request = FullPipelineWithImagesRequest(
            story_type="motivation",  # Use story type instead of script_lines
            use_ffmpeg=True,  # Use FFmpeg for video creation
            use_stable_diffusion=False,  # Use DALL-E for images
            animation_type="zoom_in",  # Animation type for video
            upload_to_s3=True,  # Enable S3 upload
            s3_folder="test_results"  # S3 folder for uploads
        )
        
        # Run the full pipeline
        result = await full_pipeline_with_images(request)
        
        end_time = time.time()
        pipeline_time = end_time - start_time
        
        if result and 'video_path' in result:
            video_path = result['video_path']
            if os.path.exists(video_path):
                print(f"✅ Full pipeline completed: {video_path}")
                print(f"   Pipeline time: {pipeline_time:.2f} seconds")
                print(f"   File size: {os.path.getsize(video_path) / 1024:.1f} KB")
                
                # Check if S3 URL is in result
                if 's3_url' in result:
                    print(f"✅ Video uploaded to S3: {result['s3_url']}")
                
                return True
            else:
                print("❌ Video file not found")
                return False
        else:
            print("❌ Full pipeline failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    
    async def run_tests():
        print("🚀 Comprehensive Fix Test Suite with S3 Upload")
        print("=" * 80)
        
        try:
            # Test image generation
            image_ok = test_image_generation()
            
            # Test motion generation
            motion_ok = test_motion_generation()
            
            # Test FFmpeg integration
            ffmpeg_ok = test_ffmpeg_integration()
            
            # Test full pipeline
            pipeline_ok = await test_full_pipeline()
            
            print("\n📊 Test Results:")
            print(f"   Image Generation: {'✅' if image_ok else '❌'}")
            print(f"   Motion Generation: {'✅' if motion_ok else '❌'}")
            print(f"   FFmpeg Integration: {'✅' if ffmpeg_ok else '❌'}")
            print(f"   Full Pipeline: {'✅' if pipeline_ok else '❌'}")
            
            if image_ok and motion_ok and ffmpeg_ok and pipeline_ok:
                print("\n🎉 All tests passed! Both image and motion generation should work.")
                print("📤 Generated files have been uploaded to S3 for easy access.")
            else:
                print("\n⚠️  Some tests failed. Check the issues above.")
                
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Run the async tests
    asyncio.run(run_tests()) 