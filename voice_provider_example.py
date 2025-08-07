#!/usr/bin/env python3
"""
Voice Provider Switching Example

This example demonstrates how to switch between different voice synthesis providers:
- ElevenLabs (cloud-based, high quality)
- Coqui TTS Bark (local, open-source)

Usage: python voice_provider_example.py
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demonstrate_provider_switching():
    """Demonstrate switching between voice synthesis providers"""
    logger.info("🎤 Voice Provider Switching Demo")
    logger.info("=" * 60)
    
    try:
        from utils.unified_voice_synthesizer import UnifiedVoiceSynthesizer
        
        # Test text
        test_lines = [
            "Hello, this is a test of voice synthesis.",
            "We can switch between different providers easily."
        ]
        
        # Initialize with Coqui TTS (default)
        logger.info("🎯 Starting with Coqui TTS...")
        synthesizer = UnifiedVoiceSynthesizer(
            provider="coqui",
            config={
                "gpu": True,
                "speaker": "random",
                "voice_dir": "bark_voices/"
            }
        )
        
        # Generate audio with Coqui TTS
        coqui_output = "test_coqui_switch.wav"
        coqui_file = synthesizer.synthesize_voice(test_lines, coqui_output)
        
        if coqui_file:
            logger.info(f"✅ Coqui TTS audio: {coqui_file}")
        
        # Get provider info
        coqui_info = synthesizer.get_provider_info()
        logger.info(f"Coqui info: {coqui_info}")
        
        # Switch to ElevenLabs (if available)
        try:
            logger.info("\n🔄 Switching to ElevenLabs...")
            synthesizer.switch_provider("elevenlabs")
            
            # Generate audio with ElevenLabs
            elevenlabs_output = "test_elevenlabs_switch.wav"
            elevenlabs_file = synthesizer.synthesize_voice(test_lines, elevenlabs_output)
            
            if elevenlabs_file:
                logger.info(f"✅ ElevenLabs audio: {elevenlabs_file}")
            
            # Get provider info
            elevenlabs_info = synthesizer.get_provider_info()
            logger.info(f"ElevenLabs info: {elevenlabs_info}")
            
        except Exception as e:
            logger.warning(f"⚠️ ElevenLabs not available: {e}")
        
        # Switch back to Coqui TTS
        logger.info("\n🔄 Switching back to Coqui TTS...")
        synthesizer.switch_provider("coqui", {
            "gpu": True,
            "speaker": "random"
        })
        
        # Generate another audio with Coqui TTS
        coqui_output2 = "test_coqui_switch2.wav"
        coqui_file2 = synthesizer.synthesize_voice(test_lines, coqui_output2)
        
        if coqui_file2:
            logger.info(f"✅ Coqui TTS audio 2: {coqui_file2}")
        
        # Cleanup
        synthesizer.cleanup()
        
        logger.info("\n🎉 Provider switching demo completed!")
        
    except Exception as e:
        logger.error(f"❌ Provider switching demo failed: {e}")

def demonstrate_voice_cloning():
    """Demonstrate voice cloning with Coqui TTS"""
    logger.info("\n🎭 Voice Cloning Demo (Coqui TTS)")
    logger.info("=" * 60)
    
    try:
        from utils.unified_voice_synthesizer import UnifiedVoiceSynthesizer
        
        # Initialize Coqui TTS synthesizer
        synthesizer = UnifiedVoiceSynthesizer(
            provider="coqui",
            config={
                "gpu": True,
                "speaker": "random",
                "voice_dir": "bark_voices/"
            }
        )
        
        # Example: Clone a voice from an audio file
        # Note: You would need to provide an actual audio file
        audio_file_path = "sample_voice.wav"  # Replace with actual file
        
        if os.path.exists(audio_file_path):
            logger.info(f"Cloning voice from: {audio_file_path}")
            
            success = synthesizer.clone_voice(
                audio_file_path=audio_file_path,
                speaker_name="cloned_voice",
                test_text="This is a test of the cloned voice."
            )
            
            if success:
                logger.info("✅ Voice cloning successful!")
                
                # Use the cloned voice
                test_lines = ["Hello, this is the cloned voice speaking."]
                cloned_output = "test_cloned_voice.wav"
                
                cloned_file = synthesizer.synthesize_voice(
                    test_lines, 
                    cloned_output,
                    speaker="cloned_voice"
                )
                
                if cloned_file:
                    logger.info(f"✅ Cloned voice audio: {cloned_file}")
            else:
                logger.error("❌ Voice cloning failed")
        else:
            logger.info("ℹ️ No sample audio file found. Skipping voice cloning demo.")
            logger.info("To test voice cloning, provide an audio file and update the path.")
        
        # Cleanup
        synthesizer.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Voice cloning demo failed: {e}")

def demonstrate_kids_voice():
    """Demonstrate voice synthesis for kids content"""
    logger.info("\n👶 Kids Voice Demo")
    logger.info("=" * 60)
    
    try:
        from utils.unified_voice_synthesizer import UnifiedVoiceSynthesizer
        
        # Kids-friendly text
        kids_lines = [
            "Hello little friends!",
            "Let's learn something fun today!",
            "Are you ready for an adventure?"
        ]
        
        # Initialize with Coqui TTS for kids content
        synthesizer = UnifiedVoiceSynthesizer(
            provider="coqui",
            config={
                "gpu": True,
                "speaker": "random",
                "voice_dir": "bark_voices/",
                "text_temp": 0.8,  # Slightly more expressive for kids
                "waveform_temp": 0.7
            }
        )
        
        # Generate kids voice
        kids_output = "kids_voice_demo.wav"
        kids_file = synthesizer.synthesize_voice(kids_lines, kids_output)
        
        if kids_file:
            logger.info(f"✅ Kids voice generated: {kids_file}")
        
        # Get available speakers
        speakers = synthesizer.get_available_speakers()
        logger.info(f"Available speakers: {speakers}")
        
        # Cleanup
        synthesizer.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Kids voice demo failed: {e}")

def show_provider_comparison():
    """Show detailed comparison of voice providers"""
    logger.info("\n📊 Voice Provider Comparison")
    logger.info("=" * 60)
    
    from utils.unified_voice_synthesizer import compare_providers
    compare_providers()

if __name__ == "__main__":
    # Show provider comparison
    show_provider_comparison()
    
    # Demonstrate provider switching
    demonstrate_provider_switching()
    
    # Demonstrate voice cloning
    demonstrate_voice_cloning()
    
    # Demonstrate kids voice
    demonstrate_kids_voice() 