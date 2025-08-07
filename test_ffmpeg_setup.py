#!/usr/bin/env python3
"""
FFmpeg Setup Test

This script tests if FFmpeg and related dependencies are working properly.
"""

import os
import sys
import logging
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ffmpeg_binary():
    """Test if FFmpeg binary is available"""
    logger.info("🔍 Testing FFmpeg binary...")
    
    try:
        # Check if ffmpeg is in PATH
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("✅ FFmpeg binary found and working")
            logger.info(f"Version: {result.stdout.split('ffmpeg version')[1].split()[0]}")
            return True
        else:
            logger.error("❌ FFmpeg binary found but not working")
            return False
            
    except FileNotFoundError:
        logger.error("❌ FFmpeg binary not found in PATH")
        logger.info("💡 Please install FFmpeg:")
        logger.info("   Windows: Download from https://ffmpeg.org/download.html")
        logger.info("   macOS: brew install ffmpeg")
        logger.info("   Ubuntu: sudo apt install ffmpeg")
        return False
    except Exception as e:
        logger.error(f"❌ FFmpeg test failed: {e}")
        return False

def test_ffmpeg_python():
    """Test if ffmpeg-python package is working"""
    logger.info("🔍 Testing ffmpeg-python package...")
    
    try:
        import ffmpeg
        logger.info("✅ ffmpeg-python package imported successfully")
        
        # Test basic ffmpeg-python functionality
        probe = ffmpeg.probe('test.mp4', quiet=True)
        logger.info("✅ ffmpeg-python probe test passed")
        return True
        
    except ImportError as e:
        logger.error(f"❌ ffmpeg-python import failed: {e}")
        logger.info("💡 Try: pip install ffmpeg-python")
        return False
    except Exception as e:
        logger.error(f"❌ ffmpeg-python test failed: {e}")
        return False

def test_moviepy():
    """Test if moviepy is working"""
    logger.info("🔍 Testing moviepy...")
    
    try:
        from moviepy.editor import VideoFileClip
        logger.info("✅ moviepy imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ moviepy import failed: {e}")
        logger.info("💡 Try: pip install moviepy")
        return False
    except Exception as e:
        logger.error(f"❌ moviepy test failed: {e}")
        return False

def test_opencv():
    """Test if OpenCV is working"""
    logger.info("🔍 Testing OpenCV...")
    
    try:
        import cv2
        logger.info(f"✅ OpenCV imported successfully (version: {cv2.__version__})")
        return True
        
    except ImportError as e:
        logger.error(f"❌ OpenCV import failed: {e}")
        logger.info("💡 Try: pip install opencv-python")
        return False
    except Exception as e:
        logger.error(f"❌ OpenCV test failed: {e}")
        return False

def test_video_processing():
    """Test basic video processing capabilities"""
    logger.info("🔍 Testing video processing...")
    
    try:
        import ffmpeg
        
        # Create a simple test video using ffmpeg
        test_output = "test_output.mp4"
        
        # Generate a simple test video
        (
            ffmpeg
            .input('testsrc=duration=2:size=320x240:rate=1', f='lavfi')
            .output(test_output, vcodec='libx264', acodec='aac')
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        
        if os.path.exists(test_output):
            logger.info("✅ Video processing test successful")
            os.remove(test_output)  # Clean up
            return True
        else:
            logger.error("❌ Video processing test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Video processing test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 FFmpeg Setup Test")
    logger.info("=" * 50)
    
    tests = [
        ("FFmpeg Binary", test_ffmpeg_binary),
        ("ffmpeg-python Package", test_ffmpeg_python),
        ("MoviePy", test_moviepy),
        ("OpenCV", test_opencv),
        ("Video Processing", test_video_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🎯 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary")
    logger.info("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎉 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! FFmpeg setup is working correctly.")
    else:
        logger.info("⚠️ Some tests failed. Please check the errors above.")
        logger.info("💡 Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 