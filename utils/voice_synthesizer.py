import os
import requests
import json
from typing import List
from pydantic import BaseModel
import tempfile

class VoiceSynthesizer:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Default calm male voice
        self.base_url = "https://api.elevenlabs.io/v1"
    
    def synthesize_voice(self, narration_lines: List[str], output_path: str) -> str:
        """Synthesize voice for narration lines using ElevenLabs API"""
        
        # Combine all narration lines into one text
        full_text = " ".join(narration_lines)
        
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
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            
            # Save the audio file
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            return output_path
            
        except requests.exceptions.RequestException as e:
            # Fallback: create a silent audio file
            return self._create_silent_audio(output_path, len(narration_lines) * 3)
    
    def _create_silent_audio(self, output_path: str, duration_seconds: int) -> str:
        """Create a silent audio file as fallback"""
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
        
        return output_path
    
    def get_available_voices(self) -> List[dict]:
        """Get list of available voices from ElevenLabs"""
        url = f"{self.base_url}/voices"
        headers = {"xi-api-key": self.api_key}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()["voices"]
        except:
            return [] 