#!/usr/bin/env python3
"""
Test script for XTTS v2 model specifically
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

def test_xtts_v2():
    """Test XTTS v2 model specifically"""
    logger.info("Testing XTTS v2 model...")

    try:
        # Apply the patch first
        import utils.pytorch_patch
        logger.info("✅ PyTorch patch imported successfully")

        # Import TTS
        from TTS.api import TTS
        logger.info("✅ TTS imported successfully")

        # Test XTTS v2 model
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        logger.info(f"Testing model: {model_name}")
        
        # Get device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load the model
        tts = TTS(model_name).to(device)
        logger.info(f"✅ Successfully loaded XTTS v2 model")

        # Test basic functionality
        logger.info("Testing basic XTTS v2 functionality...")
        test_text = "Hello, this is a test of the XTTS v2 model."

        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name

        # Create a simple speaker audio file for testing
        speaker_audio_path = "test_speaker.wav"
        create_test_speaker_audio(speaker_audio_path)

        try:
            # Test XTTS v2 synthesis
            tts.tts_to_file(
                text=test_text,
                speaker_wav=speaker_audio_path,
                language="en",
                file_path=output_path
            )
            logger.info(f"✅ XTTS v2 synthesis successful: {output_path}")

            # Check if file was created and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info("✅ Audio file created successfully with content")
                logger.info(f"🎉 SUCCESS: XTTS v2 model is working correctly!")
                
                # Clean up
                try:
                    os.unlink(output_path)
                    os.unlink(speaker_audio_path)
                except:
                    pass
                
                return True
            else:
                logger.warning("⚠️ Audio file was created but appears to be empty")

        except Exception as e:
            logger.error(f"❌ XTTS v2 synthesis failed: {e}")
            # Clean up
            try:
                os.unlink(output_path)
                os.unlink(speaker_audio_path)
            except:
                pass

        return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def create_test_speaker_audio(output_path: str):
    """Create a simple test speaker audio file"""
    import wave
    import struct
    import math
    
    sample_rate = 22050
    duration = 3.0  # 3 seconds
    frequency = 440  # A4 note
    amplitude = 0.3
    
    num_samples = int(sample_rate * duration)
    
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Generate sine wave data
        audio_data = []
        for i in range(num_samples):
            sample = amplitude * math.sin(2 * math.pi * frequency * i / sample_rate)
            audio_data.append(struct.pack('<h', int(sample * 32767)))
        
        wav_file.writeframes(b''.join(audio_data))
    
    logger.info(f"Test speaker audio created: {output_path}")

if __name__ == "__main__":
    success = test_xtts_v2()
    sys.exit(0 if success else 1) 