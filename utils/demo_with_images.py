#!/usr/bin/env python3
"""
Demonstration of complete flow with DALL-E images, S3 upload, and Shotstack video creation
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

def demonstrate_complete_flow_with_images():
    """
    Demonstrate the complete flow: Script → Voice (S3) → Images (S3) → Video (Shotstack)
    """
    print("🎬 Complete Flow with DALL-E Images and S3 Integration")
    print("=" * 60)
    
    try:
        # Step 1: Initialize utilities
        print("\n1. Initializing utilities...")
        from script_generator import ScriptGenerator
        from voice_synthesizer import VoiceSynthesizer
        from image_generator import ImageGenerator
        from shotstack_video_creator import ShotstackVideoCreator
        
        script_gen = ScriptGenerator()
        voice_synth = VoiceSynthesizer()
        image_gen = ImageGenerator()
        video_creator = ShotstackVideoCreator()
        print("✅ All utilities initialized successfully")
        
        # Step 2: Generate script
        print("\n2. Generating script...")
        script = script_gen.generate_script()
        narration_texts = [line.text for line in script.narration]
        print(f"✅ Script generated: {script.title}")
        print(f"   Narration lines: {len(narration_texts)}")
        
        # Step 3: Generate voice and upload to S3
        print("\n3. Generating voice and uploading to S3...")
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir="output") as tmp_file:
            audio_path = tmp_file.name
        
        s3_audio_url = voice_synth.synthesize_and_upload_to_s3(narration_texts, audio_path)
        if not s3_audio_url:
            print("❌ Failed to generate voice or upload to S3")
            return False
        
        print(f"✅ Voice generated and uploaded to S3")
        print(f"🔗 S3 Audio URL: {s3_audio_url}")
        
        # Step 4: Generate images and upload to S3
        print("\n4. Generating images with DALL-E and uploading to S3...")
        image_urls = image_gen.generate_images_for_script(narration_texts, "relatable")
        successful_image_urls = [url for url in image_urls if url is not None]
        
        print(f"✅ Generated {len(successful_image_urls)} images")
        for i, url in enumerate(successful_image_urls):
            print(f"   Image {i+1}: {url}")
        
        # Step 5: Create video with images using Shotstack
        print("\n5. Creating video with Shotstack using S3 audio and images...")
        
        # Convert script lines to narration format
        narration_lines = []
        current_time = 0.0
        for line in script.narration:
            duration = line.duration if hasattr(line, 'duration') else 3.0
            narration_lines.append({
                "text": line.text,
                "duration": duration,
                "start": current_time
            })
            current_time += duration
        
        # Create output video path
        timestamp = int(os.path.getmtime(audio_path))
        video_filename = f"complete_demo_{timestamp}.mp4"
        video_path = os.path.join("output", video_filename)
        
        # Create video with images
        result_video_path = video_creator.create_video_with_images(
            s3_audio_url, 
            narration_lines, 
            successful_image_urls, 
            video_path
        )
        
        print(f"✅ Video with images created successfully")
        print(f"🎬 Video path: {result_video_path}")
        
        # Step 6: Show complete flow summary
        print("\n6. Complete Flow Summary:")
        print("   Script Generation → Voice Synthesis → S3 Audio Upload → DALL-E Image Generation → S3 Image Upload → Shotstack Video Creation")
        print(f"   📝 Script: {script.title}")
        print(f"   🔗 S3 Audio: {s3_audio_url}")
        print(f"   🖼️  S3 Images: {len(successful_image_urls)} images")
        print(f"   🎬 Final Video: {result_video_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        return False

def test_image_generation_only():
    """
    Test only the image generation functionality
    """
    print("\n🖼️ Testing DALL-E Image Generation Only")
    print("=" * 40)
    
    try:
        from image_generator import ImageGenerator
        
        image_gen = ImageGenerator()
        
        # Test with sample text
        test_texts = [
            "A person looking confused at their phone",
            "Someone trying to figure out a complicated remote control",
            "A person staring at a computer screen with a puzzled expression"
        ]
        
        print(f"Generating {len(test_texts)} test images...")
        image_urls = image_gen.generate_images_for_script(test_texts, "relatable")
        
        successful_urls = [url for url in image_urls if url is not None]
        print(f"✅ Generated {len(successful_urls)} images successfully")
        
        for i, url in enumerate(successful_urls):
            print(f"   Image {i+1}: {url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing image generation: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Complete Flow with Images Demonstration")
    print("Make sure your .env file has all required API keys:")
    print("- OPENAI_API_KEY (for script generation and DALL-E)")
    print("- ELEVENLABS_API_KEY (for voice synthesis)")
    print("- SHOTSTACK_API_KEY (for video creation)")
    print("- AWS credentials (for S3 upload)")
    print()
    
    # Test image generation only
    test_image_generation_only()
    
    # Demonstrate complete flow
    demonstrate_complete_flow_with_images() 