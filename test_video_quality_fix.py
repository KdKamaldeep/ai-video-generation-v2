#!/usr/bin/env python3
"""
Test script for the improved video quality and combination fixes
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
    """Test the improved video quality and combination fixes"""
    logger.info("Testing improved video quality and combination fixes...")

    try:
        # Import the kids cartoon generator
        from kids_cartoon_generator import KidsCartoonGenerator
        logger.info("✅ Kids cartoon generator imported successfully")

        # Initialize the generator
        generator = KidsCartoonGenerator()
        logger.info("✅ Kids cartoon generator initialized")

        # Test with a shorter duration for testing
        logger.info("Testing with 20-second educational cartoon...")
        result = generator.generate_kids_cartoon(
            story_type="educational",
            cartoon_style="cute",
            duration_seconds=20,  # Shorter for testing
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
    
    # Then test the video quality fixes
    success = test_video_quality_fix()
    sys.exit(0 if success else 1) 