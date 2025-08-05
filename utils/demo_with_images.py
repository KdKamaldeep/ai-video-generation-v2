#!/usr/bin/env python3
"""
Demonstration of complete flow with DALL-E images and local file storage
"""

import os
import tempfile
import logging
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_complete_flow_with_images():
    """
    Demonstrate the complete flow: Script → Voice → Images → Video (Shotstack)
    """
    print("🎬 Complete Flow with DALL-E Images and Local Storage")
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
        
        # Step 3: Generate voice
        print("\n3. Generating voice...")
        timestamp = int(time.time())
        audio_filename = f"voice_{timestamp}.mp3"
        audio_path = os.path.join("output", audio_filename)
        
        voice_synth.synthesize_voice(narration_texts, audio_path)
        print(f"✅ Voice generated")
        print(f"🔗 Audio path: {audio_path}")
        
        # Step 4: Generate images
        print("\n4. Generating images with DALL-E...")
        image_paths = image_gen.generate_images_for_script(narration_texts, "relatable")
        successful_image_paths = [path for path in image_paths if path is not None]
        
        print(f"✅ Generated {len(successful_image_paths)} images")
        for i, path in enumerate(successful_image_paths):
            print(f"   Image {i+1}: {path}")
        
        # Step 5: Create video with images using Shotstack
        print("\n5. Creating video with Shotstack using local audio and images...")
        
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
        video_filename = f"complete_demo_{timestamp}.mp4"
        video_path = os.path.join("output", video_filename)
        
        # Create video with images
        result_video_path = video_creator.create_video_with_images(
            audio_path, 
            narration_lines, 
            successful_image_paths, 
            video_path
        )
        
        print(f"✅ Video with images created successfully")
        print(f"🎬 Video path: {result_video_path}")
        
        # Step 6: Show complete flow summary
        print("\n6. Complete Flow Summary:")
        print("   Script Generation → Voice Synthesis → DALL-E Image Generation → Shotstack Video Creation")
        print(f"   📝 Script: {script.title}")
        print(f"   🔗 Audio: {audio_path}")
        print(f"   🖼️  Images: {len(successful_image_paths)} images")
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
        image_paths = image_gen.generate_images_for_script(test_texts, "relatable")
        
        successful_paths = [path for path in image_paths if path is not None]
        print(f"✅ Generated {len(successful_paths)} images successfully")
        
        for i, path in enumerate(successful_paths):
            print(f"   Image {i+1}: {path}")
        
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
    print()
    
    # Test image generation only
    test_image_generation_only()
    
    # Demonstrate complete flow
    demonstrate_complete_flow_with_images() 