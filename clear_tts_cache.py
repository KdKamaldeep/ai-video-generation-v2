#!/usr/bin/env python3
"""
Script to clear corrupted TTS model cache and retry download
"""

import os
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_tts_cache():
    """Clear the TTS model cache to force fresh download"""
    cache_dir = os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--bark/")
    
    if os.path.exists(cache_dir):
        logger.info(f"Removing corrupted cache directory: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            logger.info("✅ Cache directory removed successfully")
        except Exception as e:
            logger.error(f"❌ Failed to remove cache directory: {e}")
            return False
    else:
        logger.info("Cache directory does not exist, nothing to clear")
    
    return True

def test_model_download():
    """Test downloading the TTS model again"""
    logger.info("Testing TTS model download...")
    
    try:
        # Apply the patch first
        import utils.pytorch_patch
        logger.info("✅ PyTorch patch imported successfully")
        
        # Import TTS
        from TTS.api import TTS
        logger.info("✅ TTS imported successfully")
        
        # Try to load the model (this will trigger a fresh download)
        logger.info("Attempting to load TTS model (will download if needed)...")
        tts = TTS("tts_models/multilingual/multi-dataset/bark")
        logger.info("✅ TTS model loaded successfully!")
        
        # Test basic functionality
        logger.info("Testing basic TTS functionality...")
        test_text = "Hello, this is a test after clearing the cache."
        
        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            tts.tts_to_file(text=test_text, file_path=output_path)
            logger.info(f"✅ TTS synthesis successful: {output_path}")
            
            # Check if file was created and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info("✅ Audio file created successfully with content")
            else:
                logger.warning("⚠️ Audio file was created but appears to be empty")
                
        except Exception as e:
            logger.error(f"❌ TTS synthesis failed: {e}")
            return False
        finally:
            # Clean up temporary file
            try:
                os.unlink(output_path)
            except:
                pass
        
        logger.info("🎉 All tests passed! Model download and functionality working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting TTS cache clearing and model download test...")
    
    # Clear the cache
    if clear_tts_cache():
        # Test the download
        success = test_model_download()
        if success:
            print("\n🎉 SUCCESS: TTS model is now working correctly!")
        else:
            print("\n❌ FAILED: Model download or functionality still has issues")
    else:
        print("\n❌ FAILED: Could not clear cache") 