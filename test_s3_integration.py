#!/usr/bin/env python3
"""
Test script for S3 integration with ElevenLabs and Shotstack
"""

import os
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API base URL
BASE_URL = "http://localhost:8000"

def test_s3_integration():
    """Test the complete S3 integration workflow"""
    
    print("🚀 Testing S3 Integration with ElevenLabs and Shotstack")
    print("=" * 60)
    
    # Test 1: Check API status
    print("\n1. Checking API status...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is running: {data['message']}")
            print(f"📦 S3 Integration: {'✅ Enabled' if data['s3_integration']['enabled'] else '❌ Disabled'}")
            if data['s3_integration']['enabled']:
                print(f"🪣 S3 Bucket: {data['s3_integration']['bucket']}")
        else:
            print(f"❌ API is not responding: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        return False
    
    # Test 2: Generate voice with S3 upload
    print("\n2. Testing voice generation with S3 upload...")
    test_narration = [
        "Welcome to Why Would You!",
        "Today we're testing our new S3 integration.",
        "This audio will be uploaded to AWS S3 and used with Shotstack."
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/generate-voice-s3",
            json={"narration": test_narration}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Voice generated and uploaded to S3")
            print(f"🔗 S3 URL: {data['audio_url']}")
            print(f"📁 Local path: {data['local_path']}")
            audio_url = data['audio_url']
        else:
            print(f"❌ Voice generation failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Voice generation error: {e}")
        return False
    
    # Test 3: Create video with S3 audio
    print("\n3. Testing video creation with S3 audio...")
    narration_lines = [
        {"text": "Welcome to Why Would You!", "duration": 3.0},
        {"text": "Today we're testing our new S3 integration.", "duration": 4.0},
        {"text": "This audio will be uploaded to AWS S3 and used with Shotstack.", "duration": 5.0}
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/create-video-s3",
            json={
                "audio_url": audio_url,
                "narration_lines": narration_lines
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Video created successfully")
            print(f"🎬 Video path: {data['video_path']}")
            print(f"🔗 Audio URL used: {data['audio_url']}")
        else:
            print(f"❌ Video creation failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Video creation error: {e}")
        return False
    
    # Test 4: Full S3 pipeline
    print("\n4. Testing complete S3 pipeline...")
    try:
        response = requests.post(f"{BASE_URL}/full-pipeline-s3")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Full S3 pipeline completed successfully")
            print(f"📝 Script title: {data['script']['title']}")
            print(f"🔗 S3 Audio URL: {data['audio_url']}")
            print(f"🎬 Video path: {data['video_path']}")
        else:
            print(f"❌ Full pipeline failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Full pipeline error: {e}")
        return False
    
    print("\n🎉 All S3 integration tests completed successfully!")
    return True

def test_s3_uploader_directly():
    """Test S3 uploader directly"""
    print("\n🔧 Testing S3 Uploader directly...")
    
    try:
        from utils.s3_uploader import S3Uploader
        
        # Create a test file
        test_content = "This is a test file for S3 upload"
        test_file_path = "test_upload.txt"
        
        with open(test_file_path, "w") as f:
            f.write(test_content)
        
        # Test upload
        s3_uploader = S3Uploader()
        s3_url = s3_uploader.upload_file_with_custom_key(
            test_file_path, 
            "test/test_upload.txt", 
            "text/plain"
        )
        
        if s3_url:
            print(f"✅ Direct S3 upload successful: {s3_url}")
            
            # Test listing files
            files = s3_uploader.list_files("test/")
            print(f"📁 Files in test/ directory: {files}")
            
            # Clean up test file
            os.remove(test_file_path)
            return True
        else:
            print("❌ Direct S3 upload failed")
            return False
            
    except Exception as e:
        print(f"❌ Direct S3 uploader test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 S3 Integration Test Suite")
    print("Make sure your API server is running on localhost:8000")
    print("Make sure you have set up your AWS credentials in .env file")
    print()
    
    # Test direct S3 uploader
    test_s3_uploader_directly()
    
    # Test API integration
    test_s3_integration() 