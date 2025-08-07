#!/usr/bin/env python3
"""
Test script for alternative TTS models to avoid the corrupted bark model
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

def test_alternative_models():
    """Test with alternative TTS models"""
    logger.info("Testing alternative TTS models...")

    try:
        # Apply the patch first
        import utils.pytorch_patch
        logger.info("✅ PyTorch patch imported successfully")

        # Import TTS
        from TTS.api import TTS
        logger.info("✅ TTS imported successfully")

        # List available models
        logger.info("Available TTS models:")
        models = TTS.list_models()
        for i, model in enumerate(models[:10]):  # Show first 10 models
            logger.info(f"  {i+1}. {model}")

        # Try a different model - let's use a simpler one
        alternative_models = [
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/fast_pitch",
            "tts_models/en/vctk/vits",
            "tts_models/multilingual/multi-dataset/your_tts"
        ]

        for model_name in alternative_models:
            logger.info(f"\nTrying model: {model_name}")
            try:
                tts = TTS(model_name)
                logger.info(f"✅ Successfully loaded model: {model_name}")

                # Test basic functionality
                logger.info("Testing basic TTS functionality...")
                test_text = "Hello, this is a test of the alternative model."

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
                        logger.info(f"🎉 SUCCESS: Model {model_name} is working correctly!")
                        
                        # Clean up
                        try:
                            os.unlink(output_path)
                        except:
                            pass
                        
                        return True
                    else:
                        logger.warning("⚠️ Audio file was created but appears to be empty")

                except Exception as e:
                    logger.error(f"❌ TTS synthesis failed: {e}")
                    # Clean up
                    try:
                        os.unlink(output_path)
                    except:
                        pass

            except Exception as e:
                logger.error(f"❌ Failed to load model {model_name}: {e}")
                continue

        logger.error("❌ All alternative models failed")
        return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_alternative_models()
    sys.exit(0 if success else 1) 