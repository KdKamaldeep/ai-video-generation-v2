import os
import requests
import json
import logging
from typing import List, Optional
from pydantic import BaseModel
import tempfile

logger = logging.getLogger(__name__)

class VoiceSynthesizer:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.error("ELEVENLABS_API_KEY environment variable is not set")
            raise ValueError("ELEVENLABS_API_KEY environment variable is not set")
        
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Default calm male voice
        self.base_url = "https://api.elevenlabs.io/v1"
        logger.info(f"Initializing ElevenLabs client with voice ID: {self.voice_id}")
    
    def synthesize_voice(self, narration_lines: List[str], output_path: str) -> str:
        """Synthesize voice for narration lines using ElevenLabs API"""
        
        logger.info(f"Synthesizing voice for {len(narration_lines)} lines")
        
        # Combine all narration lines into one text
        full_text = " ".join(narration_lines)
        logger.info(f"Combined text length: {len(full_text)} characters")
        
        # Prepare the API request
        url = f"{self.base_url}/text-to-speech/{self.voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": full_text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        try:
            logger.info("Sending request to ElevenLabs API")
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            
            # Save the audio file
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Voice synthesized successfully: {output_path}")
            return output_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to synthesize voice with ElevenLabs: {str(e)}")
            logger.info("Creating fallback silent audio")
            # Fallback: create a silent audio file
            return self._create_silent_audio(output_path, len(narration_lines) * 3)
    
    def synthesize_and_upload_to_s3(self, narration_lines: List[str], output_path: str) -> Optional[str]:
        """
        Synthesize voice and upload to S3, returning the public URL
        
        Args:
            narration_lines: List of text lines to synthesize
            output_path: Local path to save the audio file temporarily
            
        Returns:
            Public S3 URL of the uploaded audio file, or None if failed
        """
        try:
            # First synthesize the voice locally
            local_path = self.synthesize_voice(narration_lines, output_path)
            
            # Import S3Uploader here to avoid circular imports
            from .s3_uploader import S3Uploader
            
            # Upload to S3
            s3_uploader = S3Uploader()
            s3_url = s3_uploader.upload_audio_file(local_path)
            
            if s3_url:
                logger.info(f"Audio successfully uploaded to S3: {s3_url}")
                return s3_url
            else:
                logger.error("Failed to upload audio to S3")
                return None
                
        except Exception as e:
            logger.error(f"Error in synthesize_and_upload_to_s3: {e}")
            return None
    
    def _create_silent_audio(self, output_path: str, duration_seconds: int) -> str:
        """Create a silent audio file as fallback"""
        logger.info(f"Creating silent audio file: {output_path} ({duration_seconds}s)")
        import wave
        import struct
        
        # Create a silent WAV file
        sample_rate = 44100
        num_samples = int(sample_rate * duration_seconds)
        
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Generate silent audio data
            silent_data = struct.pack('<h', 0) * num_samples
            wav_file.writeframes(silent_data)
        
        logger.info("Silent audio file created successfully")
        return output_path
    
    def get_available_voices(self) -> List[dict]:
        """Get list of available voices from ElevenLabs"""
        logger.info("Fetching available voices from ElevenLabs")
        url = f"{self.base_url}/voices"
        headers = {"xi-api-key": self.api_key}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            voices = response.json()["voices"]
            logger.info(f"Retrieved {len(voices)} voices from ElevenLabs")
            return voices
        except Exception as e:
            logger.error(f"Failed to get available voices: {str(e)}")
            return [] 