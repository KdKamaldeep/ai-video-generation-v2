#!/usr/bin/env python3
"""
Test script for PyTorch 2.6 compatibility patch
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

def test_pytorch_patch():
    """Test the PyTorch 2.6 compatibility patch"""
    logger.info("Testing PyTorch 2.6 compatibility patch...")
    
    try:
        # Apply the patch first
        import utils.pytorch_patch
        logger.info("✅ PyTorch patch imported successfully")
        
        # Test importing TTS
        logger.info("Testing TTS import...")
        from TTS.api import TTS
        logger.info("✅ TTS imported successfully")
        
        # Test model loading (this is where the error was occurring)
        logger.info("Testing TTS model loading...")
        tts = TTS("tts_models/multilingual/multi-dataset/bark")
        logger.info("✅ TTS model loaded successfully!")
        
        # Test basic functionality
        logger.info("Testing basic TTS functionality...")
        test_text = "Hello, this is a test of the PyTorch 2.6 patch."
        
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
        
        logger.info("🎉 All tests passed! PyTorch 2.6 patch is working correctly.")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("Make sure TTS is installed: pip install TTS")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_pytorch_patch()
    sys.exit(0 if success else 1) 