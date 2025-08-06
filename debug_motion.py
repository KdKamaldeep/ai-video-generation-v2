#!/usr/bin/env python3
"""
Debug script to test motion generation step by step
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

def test_single_motion_video():
    """Test generating a single motion video"""
    print("🎬 Testing Single Motion Video Generation")
    print("=" * 60)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        
        # Initialize generator
        print("1. Initializing StableDiffusionGenerator...")
        generator = StableDiffusionGenerator()
        
        if generator.video_pipeline is None:
            print("❌ Video pipeline not available")
            return False
        
        print("✅ Video pipeline loaded")
        
        # Find a test image
        test_image_path = None
        if os.path.exists("output/images"):
            image_files = [f for f in os.listdir("output/images") if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                test_image_path = os.path.join("output/images", image_files[0])
                print(f"2. Using test image: {test_image_path}")
        
        if test_image_path is None:
            print("❌ No test image found")
            return False
        
        # Generate motion video
        print("3. Generating motion video...")
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
                    print(f"   Motion detected: {'Yes' if motion_score > 5.0 else 'No'}")
                    
                    if motion_score <= 5.0:
                        print("⚠️  WARNING: Very little motion detected!")
                        return False
                else:
                    print("❌ Could not read video frames")
                    return False
            else:
                print("❌ Only one frame in video - no motion possible")
                return False
            
            return True
        else:
            print("❌ Motion video generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ffmpeg_processing():
    """Test FFmpeg video processing"""
    print("\n🎥 Testing FFmpeg Video Processing")
    print("=" * 60)
    
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
            "text": "Test motion video",
            "duration": 3.0,
            "visual_suggestion": "A person walking in a park"
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

if __name__ == "__main__":
    print("🚀 Motion Generation Debug Suite")
    print("=" * 80)
    
    try:
        # Test single motion video generation
        motion_ok = test_single_motion_video()
        
        # Test FFmpeg processing
        ffmpeg_ok = test_ffmpeg_processing()
        
        print("\n📊 Debug Results:")
        print(f"   Single Motion Video: {'✅' if motion_ok else '❌'}")
        print(f"   FFmpeg Processing: {'✅' if ffmpeg_ok else '❌'}")
        
        if motion_ok and ffmpeg_ok:
            print("\n🎉 All tests passed! Motion generation should work.")
        else:
            print("\n⚠️  Some tests failed. Check the issues above.")
            
    except Exception as e:
        print(f"❌ Debug suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 