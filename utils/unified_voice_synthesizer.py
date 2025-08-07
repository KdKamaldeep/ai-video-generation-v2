#!/usr/bin/env python3
"""
Unified Voice Synthesizer

This module provides a unified interface for voice synthesis using either:
- ElevenLabs API (cloud-based, high quality)
- Coqui TTS Bark (local, open-source)

Usage:
    synthesizer = UnifiedVoiceSynthesizer(provider="coqui")  # or "elevenlabs"
    audio_file = synthesizer.synthesize_voice(text_lines, output_path)
"""

import os
import logging
from typing import List, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

class VoiceProvider(Enum):
    """Available voice synthesis providers"""
    ELEVENLABS = "elevenlabs"
    COQUI = "coqui"

class UnifiedVoiceSynthesizer:
    def __init__(self, 
                 provider: str = "coqui",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize unified voice synthesizer
        
        Args:
            provider: Voice provider ("elevenlabs" or "coqui")
            config: Provider-specific configuration
        """
        self.provider = VoiceProvider(provider.lower())
        self.config = config or {}
        
        # Initialize the selected provider
        self._synthesizer = None
        self._initialize_provider()
        
        logger.info(f"Unified Voice Synthesizer initialized with provider: {self.provider.value}")
    
    def _initialize_provider(self):
        """Initialize the selected voice synthesis provider"""
        try:
            if self.provider == VoiceProvider.ELEVENLABS:
                from utils.voice_synthesizer import VoiceSynthesizer
                self._synthesizer = VoiceSynthesizer()
                logger.info("✅ ElevenLabs voice synthesizer initialized")
                
            elif self.provider == VoiceProvider.COQUI:
                from utils.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
                
                # Create Coqui config from provided config
                coqui_config = CoquiVoiceConfig(
                    gpu=self.config.get("gpu", True),
                    speaker=self.config.get("speaker", "random"),
                    voice_dir=self.config.get("voice_dir", "bark_voices/"),
                    text_temp=self.config.get("text_temp", 0.7),
                    waveform_temp=self.config.get("waveform_temp", 0.7)
                )
                
                self._synthesizer = CoquiVoiceSynthesizer(coqui_config)
                logger.info("✅ Coqui TTS voice synthesizer initialized")
                
        except ImportError as e:
            logger.error(f"❌ Failed to import {self.provider.value} provider: {e}")
            raise ImportError(f"{self.provider.value} provider not available: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.provider.value} provider: {e}")
            raise
    
    def synthesize_voice(self, 
                        narration_lines: List[str], 
                        output_path: str,
                        **kwargs) -> str:
        """
        Synthesize voice from text using the selected provider
        
        Args:
            narration_lines: List of text lines to synthesize
            output_path: Path to save the audio file
            **kwargs: Provider-specific arguments
            
        Returns:
            Path to the created audio file
        """
        if self._synthesizer is None:
            raise RuntimeError("Voice synthesizer not initialized")
        
        try:
            logger.info(f"Synthesizing voice using {self.provider.value}...")
            
            if self.provider == VoiceProvider.ELEVENLABS:
                # ElevenLabs synthesis
                return self._synthesizer.synthesize_voice(narration_lines, output_path)
                
            elif self.provider == VoiceProvider.COQUI:
                # Coqui TTS synthesis with additional parameters
                speaker = kwargs.get("speaker")
                voice_clone_audio = kwargs.get("voice_clone_audio")
                
                return self._synthesizer.synthesize_voice(
                    narration_lines, 
                    output_path,
                    speaker=speaker,
                    voice_clone_audio=voice_clone_audio
                )
                
        except Exception as e:
            logger.error(f"❌ Voice synthesis failed with {self.provider.value}: {e}")
            raise
    
    def clone_voice(self, 
                   audio_file_path: str, 
                   speaker_name: str,
                   test_text: str = "Hello, this is a test of the cloned voice.") -> bool:
        """
        Clone a voice from an audio file (Coqui TTS only)
        
        Args:
            audio_file_path: Path to the audio file for voice cloning
            speaker_name: Name for the cloned voice
            test_text: Text to test the cloned voice
            
        Returns:
            True if voice cloning was successful
        """
        if self.provider != VoiceProvider.COQUI:
            logger.warning("Voice cloning is only available with Coqui TTS")
            return False
        
        try:
            return self._synthesizer.clone_voice(audio_file_path, speaker_name, test_text)
        except Exception as e:
            logger.error(f"❌ Voice cloning failed: {e}")
            return False
    
    def get_available_speakers(self) -> List[str]:
        """Get list of available speakers"""
        try:
            if self.provider == VoiceProvider.ELEVENLABS:
                # ElevenLabs speakers
                return self._synthesizer.get_available_voices()
                
            elif self.provider == VoiceProvider.COQUI:
                # Coqui TTS speakers
                return self._synthesizer.get_available_speakers()
                
        except Exception as e:
            logger.error(f"Failed to get available speakers: {e}")
            return []
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider"""
        info = {
            "provider": self.provider.value,
            "available": self._synthesizer is not None
        }
        
        if self.provider == VoiceProvider.ELEVENLABS:
            info.update({
                "type": "cloud",
                "api_required": True,
                "voice_cloning": False,
                "offline": False
            })
        elif self.provider == VoiceProvider.COQUI:
            info.update({
                "type": "local",
                "api_required": False,
                "voice_cloning": True,
                "offline": True
            })
        
        return info
    
    def switch_provider(self, new_provider: str, config: Optional[Dict[str, Any]] = None):
        """
        Switch to a different voice synthesis provider
        
        Args:
            new_provider: New provider ("elevenlabs" or "coqui")
            config: Provider-specific configuration
        """
        try:
            # Cleanup current provider
            if self._synthesizer and hasattr(self._synthesizer, 'cleanup'):
                self._synthesizer.cleanup()
            
            # Update provider and config
            self.provider = VoiceProvider(new_provider.lower())
            if config:
                self.config.update(config)
            
            # Initialize new provider
            self._initialize_provider()
            
            logger.info(f"✅ Switched to {self.provider.value} provider")
            
        except Exception as e:
            logger.error(f"❌ Failed to switch provider: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self._synthesizer and hasattr(self._synthesizer, 'cleanup'):
                self._synthesizer.cleanup()
            logger.info("Unified voice synthesizer cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

def compare_providers():
    """Compare the features of different voice synthesis providers"""
    logger.info("🔍 Voice Synthesis Provider Comparison")
    logger.info("=" * 60)
    
    comparison = {
        "ElevenLabs": {
            "type": "Cloud-based API",
            "quality": "Very High",
            "speed": "Fast",
            "cost": "Paid API",
            "voice_cloning": "Advanced",
            "offline": False,
            "setup": "API key required",
            "best_for": "Production, high-quality content"
        },
        "Coqui TTS": {
            "type": "Local open-source",
            "quality": "High",
            "speed": "Medium (GPU recommended)",
            "cost": "Free",
            "voice_cloning": "Basic",
            "offline": True,
            "setup": "Model download required",
            "best_for": "Development, privacy, cost-effective"
        }
    }
    
    for provider, features in comparison.items():
        logger.info(f"\n📊 {provider}:")
        for feature, value in features.items():
            logger.info(f"  • {feature}: {value}")

def test_unified_synthesizer():
    """Test the unified voice synthesizer with both providers"""
    logger.info("🧪 Testing Unified Voice Synthesizer")
    logger.info("=" * 60)
    
    test_lines = [
        "Hello, this is a test of the unified voice synthesizer.",
        "It can switch between different voice synthesis providers."
    ]
    
    # Test Coqui TTS
    try:
        logger.info("\n🎯 Testing Coqui TTS...")
        coqui_synth = UnifiedVoiceSynthesizer(
            provider="coqui",
            config={"gpu": True, "speaker": "random"}
        )
        
        coqui_output = "test_coqui_unified.wav"
        coqui_file = coqui_synth.synthesize_voice(test_lines, coqui_output)
        
        if coqui_file:
            logger.info(f"✅ Coqui TTS test successful: {coqui_file}")
        
        coqui_info = coqui_synth.get_provider_info()
        logger.info(f"Coqui info: {coqui_info}")
        
        coqui_synth.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Coqui TTS test failed: {e}")
    
    # Test ElevenLabs (if API key available)
    try:
        logger.info("\n🎯 Testing ElevenLabs...")
        elevenlabs_synth = UnifiedVoiceSynthesizer(provider="elevenlabs")
        
        elevenlabs_output = "test_elevenlabs_unified.wav"
        elevenlabs_file = elevenlabs_synth.synthesize_voice(test_lines, elevenlabs_output)
        
        if elevenlabs_file:
            logger.info(f"✅ ElevenLabs test successful: {elevenlabs_file}")
        
        elevenlabs_info = elevenlabs_synth.get_provider_info()
        logger.info(f"ElevenLabs info: {elevenlabs_info}")
        
        elevenlabs_synth.cleanup()
        
    except Exception as e:
        logger.error(f"❌ ElevenLabs test failed: {e}")

if __name__ == "__main__":
    # Compare providers
    compare_providers()
    
    # Test unified synthesizer
    test_unified_synthesizer() 