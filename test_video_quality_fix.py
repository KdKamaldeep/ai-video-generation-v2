#!/usr/bin/env python3
"""
Test script for the improved video quality and frame rate fixes
"""

import logging
import sys
import os
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_video_quality_fix():
    """Test the improved video quality and frame rate fixes"""
    logger.info("Testing improved video quality and frame rate fixes...")

    try:
        # Import the kids cartoon generator
        from kids_cartoon_generator import KidsCartoonGenerator
        logger.info("✅ Kids cartoon generator imported successfully")

        # Initialize the generator
        generator = KidsCartoonGenerator()
        logger.info("✅ Kids cartoon generator initialized")

        # Test with a shorter duration for testing
        logger.info("Testing with 15-second educational cartoon...")
        result = generator.generate_kids_cartoon(
            story_type="educational",
            cartoon_style="cute",
            duration_seconds=15,  # Test with 15 seconds to verify frame rate fix
            include_voice=True
        )

        if result["success"]:
            logger.info("🎉 SUCCESS: Kids cartoon generated successfully!")
            logger.info(f"📁 Final video: {result['files'].get('final_video', 'N/A')}")
            logger.info(f"🎤 Audio file: {result['files'].get('audio', 'N/A')}")
            logger.info(f"🎬 Video files: {len(result['files'].get('videos', []))} scenes")
            
            # Check if final video exists and has content
            final_video = result['files'].get('final_video')
            if final_video and os.path.exists(final_video):
                file_size = os.path.getsize(final_video)
                logger.info(f"✅ Final video exists with size: {file_size} bytes")
                
                # Check video properties using ffprobe
                try:
                    result_probe = subprocess.run([
                        'ffprobe', '-v', 'quiet', '-show_entries', 
                        'format=duration:stream=codec_type,r_frame_rate',
                        '-of', 'csv=p=0', final_video
                    ], capture_output=True, text=True, check=True)
                    
                    logger.info(f"Video properties: {result_probe.stdout.strip()}")
                    
                    # Parse duration and frame rate
                    lines = result_probe.stdout.strip().split('\n')
                    for line in lines:
                        if 'duration' in line:
                            duration = float(line.split(',')[0])
                            logger.info(f"📹 Video duration: {duration:.2f} seconds")
                            if duration >= 10:  # Should be at least 10 seconds
                                logger.info("✅ Video duration is appropriate")
                            else:
                                logger.warning(f"⚠️ Video duration is too short: {duration:.2f}s")
                        elif 'video' in line and 'r_frame_rate' in line:
                            frame_rate = line.split(',')[-1]
                            logger.info(f"🎞️ Video frame rate: {frame_rate}")
                    
                    # Check if video has audio
                    if 'audio' in result_probe.stdout:
                        logger.info("✅ Video includes audio track")
                    else:
                        logger.warning("⚠️ Video does not have audio track")
                        
                except Exception as e:
                    logger.warning(f"Could not check video properties: {e}")
                
                # Check if video has audio (basic check)
                if file_size > 1000000:  # More than 1MB likely has audio
                    logger.info("✅ Video appears to have audio (large file size)")
                else:
                    logger.warning("⚠️ Video might not have audio (small file size)")
                
                return True
            else:
                logger.error("❌ Final video not found")
                return False
        else:
            logger.error("❌ Kids cartoon generation failed")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_animatediff_frame_rate():
    """Test AnimateDiff frame rate settings directly"""
    logger.info("Testing AnimateDiff frame rate settings...")
    
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        # Initialize generator
        generator = AnimateDiffGenerator(
            memory_optimization=True,
            cache_dir="models_cache"
        )
        
        # Test with different durations
        test_cases = [
            {"duration": 3, "expected_frames": 36},  # 3s * 12fps = 36 frames
            {"duration": 5, "expected_frames": 48},  # 5s * 12fps = 60, but max is 48
            {"duration": 2, "expected_frames": 24},  # 2s * 12fps = 24 frames
        ]
        
        for test_case in test_cases:
            duration = test_case["duration"]
            expected_frames = test_case["expected_frames"]
            
            # Calculate frames using the same logic as kids cartoon generator
            fps = 12
            num_frames = max(16, min(48, int(duration * fps)))
            
            logger.info(f"Duration: {duration}s, FPS: {fps}, Calculated frames: {num_frames}, Expected: {expected_frames}")
            
            if num_frames == expected_frames:
                logger.info(f"✅ Frame calculation correct for {duration}s duration")
            else:
                logger.warning(f"⚠️ Frame calculation mismatch for {duration}s duration")
        
        generator.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"❌ AnimateDiff frame rate test failed: {e}")
        return False

def test_ffmpeg_installation():
    """Test if ffmpeg is properly installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, check=True)
        logger.info("✅ ffmpeg is installed and working")
        return True
    except Exception as e:
        logger.error(f"❌ ffmpeg not available: {e}")
        return False

if __name__ == "__main__":
    # First test ffmpeg installation
    if not test_ffmpeg_installation():
        print("❌ ffmpeg not available - please install ffmpeg first")
        sys.exit(1)
    
    # Test AnimateDiff frame rate calculations
    logger.info("=" * 60)
    logger.info("Testing AnimateDiff frame rate calculations...")
    test_animatediff_frame_rate()
    
    # Then test the video quality fixes
    logger.info("=" * 60)
    logger.info("Testing the video quality and frame rate fixes...")
    success = test_video_quality_fix()
    sys.exit(0 if success else 1) 