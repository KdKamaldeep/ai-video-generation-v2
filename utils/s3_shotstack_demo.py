#!/usr/bin/env python3
"""
Demonstration of S3 URL flow to Shotstack using existing utilities
"""

import os
import tempfile
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_s3_to_shotstack_flow():
    """
    Demonstrate the complete flow: ElevenLabs → S3 → Shotstack
    """
    print("🎬 Demonstrating S3 URL Flow to Shotstack")
    print("=" * 50)
    
    try:
        # Step 1: Initialize utilities
        print("\n1. Initializing utilities...")
        from voice_synthesizer import VoiceSynthesizer
        from shotstack_video_creator import ShotstackVideoCreator
        
        voice_synth = VoiceSynthesizer()
        video_creator = ShotstackVideoCreator()
        print("✅ Utilities initialized successfully")
        
        # Step 2: Generate voice and upload to S3
        print("\n2. Generating voice and uploading to S3...")
        test_narration = [
            "Welcome to Why Would You!",
            "This is a test of our S3 integration.",
            "The audio will be uploaded to AWS S3 and used by Shotstack."
        ]
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir="output") as tmp_file:
            audio_path = tmp_file.name
        
        # Generate voice and upload to S3
        s3_audio_url = voice_synth.synthesize_and_upload_to_s3(test_narration, audio_path)
        
        if not s3_audio_url:
            print("❌ Failed to generate voice or upload to S3")
            return False
        
        print(f"✅ Voice generated and uploaded to S3")
        print(f"🔗 S3 URL: {s3_audio_url}")
        
        # Step 3: Create video using S3 URL with Shotstack
        print("\n3. Creating video with Shotstack using S3 URL...")
        narration_lines = [
            {"text": "Welcome to Why Would You!", "duration": 3.0},
            {"text": "This is a test of our S3 integration.", "duration": 4.0},
            {"text": "The audio will be uploaded to AWS S3 and used by Shotstack.", "duration": 5.0}
        ]
        
        # Create output video path
        timestamp = int(os.path.getmtime(audio_path))
        video_filename = f"s3_demo_{timestamp}.mp4"
        video_path = os.path.join("output", video_filename)
        
        # Create video using S3 URL
        result_video_path = video_creator.create_video(s3_audio_url, narration_lines, video_path)
        
        print(f"✅ Video created successfully")
        print(f"🎬 Video path: {result_video_path}")
        print(f"🔗 S3 Audio URL used: {s3_audio_url}")
        
        # Step 4: Show the complete flow
        print("\n4. Complete Flow Summary:")
        print("   ElevenLabs → Local Audio File → S3 Upload → S3 URL → Shotstack → Video")
        print(f"   📁 Local audio: {audio_path}")
        print(f"   🔗 S3 URL: {s3_audio_url}")
        print(f"   🎬 Final video: {result_video_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        return False

def test_s3_url_processing():
    """
    Test how Shotstack processes S3 URLs
    """
    print("\n🔧 Testing S3 URL Processing in Shotstack")
    print("=" * 40)
    
    try:
        from shotstack_video_creator import ShotstackVideoCreator
        
        video_creator = ShotstackVideoCreator()
        
        # Test different URL types
        test_urls = [
            "https://why-would-you.s3.us-east-1.amazonaws.com/audio/test.mp3",
            "file:///path/to/local/file.mp3",
            "https://example.com/audio.mp3"
        ]
        
        for url in test_urls:
            processed_url = video_creator._process_audio_url(url)
            print(f"Input: {url}")
            print(f"Processed: {processed_url}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing URL processing: {e}")
        return False

if __name__ == "__main__":
    print("🚀 S3 to Shotstack Flow Demonstration")
    print("Make sure your .env file has AWS credentials and other API keys")
    print()
    
    # Test URL processing
    test_s3_url_processing()
    
    # Demonstrate complete flow
    demonstrate_s3_to_shotstack_flow() 