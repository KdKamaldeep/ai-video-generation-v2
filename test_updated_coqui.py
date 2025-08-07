#!/usr/bin/env python3
"""
Test script for the updated Coqui voice synthesizer
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

def test_updated_coqui():
    """Test the updated Coqui voice synthesizer"""
    logger.info("Testing updated Coqui voice synthesizer...")

    try:
        # Apply the patch first
        import utils.pytorch_patch
        logger.info("✅ PyTorch patch imported successfully")

        # Import the updated Coqui voice synthesizer
        from utils.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
        logger.info("✅ Coqui voice synthesizer imported successfully")

        # Test initialization
        logger.info("Testing Coqui voice synthesizer initialization...")
        synthesizer = CoquiVoiceSynthesizer()
        logger.info(f"✅ Coqui voice synthesizer initialized with model: {synthesizer.config.model_name}")

        # Test basic voice synthesis
        logger.info("Testing basic voice synthesis...")
        test_lines = ["Hello, this is a test of the updated Coqui voice synthesizer."]
        
        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            result = synthesizer.synthesize_voice(
                narration_lines=test_lines,
                output_path=output_path
            )
            logger.info(f"✅ Voice synthesis successful: {result}")

            # Check if file was created and has content
            if os.path.exists(result) and os.path.getsize(result) > 0:
                logger.info("✅ Audio file created successfully with content")
                logger.info(f"🎉 SUCCESS: Updated Coqui voice synthesizer is working correctly!")
                
                # Clean up
                try:
                    os.unlink(result)
                except:
                    pass
                
                return True
            else:
                logger.warning("⚠️ Audio file was created but appears to be empty")

        except Exception as e:
            logger.error(f"❌ Voice synthesis failed: {e}")
            # Clean up
            try:
                os.unlink(output_path)
            except:
                pass

        return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_updated_coqui()
    sys.exit(0 if success else 1) 