#!/usr/bin/env python3
"""
Test script for the fixed kids cartoon generator
"""

import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_kids_cartoon_fix():
    """Test the fixed kids cartoon generator"""
    logger.info("Testing fixed kids cartoon generator...")

    try:
        # Import the kids cartoon generator
        from kids_cartoon_generator import KidsCartoonGenerator
        logger.info("✅ Kids cartoon generator imported successfully")

        # Initialize the generator
        generator = KidsCartoonGenerator()
        logger.info("✅ Kids cartoon generator initialized")

        # Test with a shorter duration for testing
        logger.info("Testing with 30-second educational cartoon...")
        result = generator.generate_kids_cartoon(
            story_type="educational",
            cartoon_style="cute",
            duration_seconds=30,  # Shorter for testing
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

if __name__ == "__main__":
    success = test_kids_cartoon_fix()
    sys.exit(0 if success else 1) 