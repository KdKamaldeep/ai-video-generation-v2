#!/usr/bin/env python3
"""
Test script to check if the video pipeline is working properly
"""

import os
import sys
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_video_pipeline_loading():
    """Test if the video pipeline loads correctly"""
    print("🔧 Testing Video Pipeline Loading")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        
        # Initialize the generator
        print("Initializing StableDiffusionGenerator...")
        generator = StableDiffusionGenerator()
        
        # Check if video pipeline is loaded
        if generator.video_pipeline is not None:
            print("✅ Video pipeline loaded successfully")
            print(f"   Device: {generator.device}")
            print(f"   Pipeline type: {type(generator.video_pipeline)}")
            return True
        else:
            print("❌ Video pipeline failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        return False

def test_video_generation():
    """Test if video generation works"""
    print("\n🎬 Testing Video Generation")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        
        # Initialize the generator
        generator = StableDiffusionGenerator()
        
        if generator.video_pipeline is None:
            print("❌ Video pipeline not available")
            return False
        
        # Create a test image path (use an existing image if available)
        test_image_path = None
        
        # Look for existing images in output directory
        if os.path.exists("output/images"):
            image_files = [f for f in os.listdir("output/images") if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                test_image_path = os.path.join("output/images", image_files[0])
                print(f"Using existing image: {test_image_path}")
        
        if test_image_path is None:
            print("❌ No test image found. Please generate an image first.")
            return False
        
        # Test video generation
        print("Generating motion video...")
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
            print(f"✅ Motion video generated successfully!")
            print(f"   Path: {video_path}")
            print(f"   Generation time: {generation_time:.2f} seconds")
            print(f"   File size: {os.path.getsize(video_path) / 1024:.1f} KB")
            return True
        else:
            print("❌ Motion video generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during video generation: {e}")
        return False

def test_ffmpeg_integration():
    """Test if FFmpeg integration works"""
    print("\n🎥 Testing FFmpeg Integration")
    print("=" * 50)
    
    try:
        from utils.ffmpeg_video_creator import FFmpegVideoCreator
        
        # Initialize FFmpeg video creator
        print("Initializing FFmpegVideoCreator...")
        ffmpeg_creator = FFmpegVideoCreator(use_stable_video_diffusion=True)
        
        if ffmpeg_creator.sd_generator is not None:
            print("✅ FFmpegVideoCreator initialized with Stable Diffusion")
            print(f"   SD Generator: {type(ffmpeg_creator.sd_generator)}")
            print(f"   Video Pipeline: {ffmpeg_creator.sd_generator.video_pipeline is not None}")
            return True
        else:
            print("❌ FFmpegVideoCreator failed to initialize Stable Diffusion")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing FFmpegVideoCreator: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Video Pipeline Test Suite")
    print("=" * 60)
    
    try:
        # Test video pipeline loading
        pipeline_ok = test_video_pipeline_loading()
        
        # Test FFmpeg integration
        ffmpeg_ok = test_ffmpeg_integration()
        
        # Test video generation if pipeline is available
        if pipeline_ok:
            video_ok = test_video_generation()
        else:
            video_ok = False
        
        print("\n📊 Test Results:")
        print(f"   Video Pipeline Loading: {'✅' if pipeline_ok else '❌'}")
        print(f"   FFmpeg Integration: {'✅' if ffmpeg_ok else '❌'}")
        print(f"   Video Generation: {'✅' if video_ok else '❌'}")
        
        if pipeline_ok and ffmpeg_ok and video_ok:
            print("\n🎉 All tests passed! Motion generation should work.")
        else:
            print("\n⚠️  Some tests failed. Check the issues above.")
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        sys.exit(1) 