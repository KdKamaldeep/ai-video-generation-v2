#!/usr/bin/env python3
"""
Test script to verify the YouTube Shorts automation setup
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment():
    """Test if all required environment variables are set"""
    print("🔍 Testing environment variables...")
    
    required_vars = [
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY", 
        "YOUTUBE_CLIENT_ID",
        "YOUTUBE_CLIENT_SECRET"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        return False
    else:
        print("✅ All environment variables are set")
        return True

def test_api_connection():
    """Test if the FastAPI server is running"""
    print("\n🔍 Testing API connection...")
    
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            return True
        else:
            print(f"❌ API server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error testing API connection: {e}")
        return False

def test_script_generation():
    """Test script generation endpoint"""
    print("\n🔍 Testing script generation...")
    
    try:
        response = requests.get("http://localhost:8000/generate-script", timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("script"):
                script = data["script"]
                print(f"✅ Script generated successfully")
                print(f"   Title: {script['title']}")
                print(f"   Duration: {script['total_duration']}s")
                print(f"   Lines: {len(script['narration'])}")
                return True
            else:
                print("❌ Script generation failed - invalid response format")
                return False
        else:
            print(f"❌ Script generation failed - status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing script generation: {e}")
        return False

def test_voice_synthesis():
    """Test voice synthesis endpoint"""
    print("\n🔍 Testing voice synthesis...")
    
    try:
        test_narration = ["This is a test narration for the Why Would You channel."]
        response = requests.post(
            "http://localhost:8000/generate-voice",
            json={"narration": test_narration},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("audio_path"):
                print("✅ Voice synthesis successful")
                print(f"   Audio file: {data['audio_path']}")
                return True
            else:
                print("❌ Voice synthesis failed - invalid response format")
                return False
        else:
            print(f"❌ Voice synthesis failed - status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing voice synthesis: {e}")
        return False

def test_available_voices():
    """Test available voices endpoint"""
    print("\n🔍 Testing available voices...")
    
    try:
        response = requests.get("http://localhost:8000/available-voices", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                voices = data.get("voices", [])
                print(f"✅ Found {len(voices)} available voices")
                return True
            else:
                print("❌ Failed to get available voices")
                return False
        else:
            print(f"❌ Available voices failed - status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing available voices: {e}")
        return False

def test_channel_info():
    """Test channel info endpoint"""
    print("\n🔍 Testing channel info...")
    
    try:
        response = requests.get("http://localhost:8000/channel-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("channel"):
                channel = data["channel"]
                print(f"✅ Channel info retrieved")
                print(f"   Channel: {channel['title']}")
                print(f"   Videos: {channel['video_count']}")
                return True
        elif response.status_code == 404:
            print("⚠️  Channel info not found (authentication may be required)")
            return True  # Not a failure, just needs auth
        else:
            print(f"❌ Channel info failed - status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing channel info: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 YouTube Shorts Automation - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Environment Variables", test_environment),
        ("API Connection", test_api_connection),
        ("Script Generation", test_script_generation),
        ("Voice Synthesis", test_voice_synthesis),
        ("Available Voices", test_available_voices),
        ("Channel Info", test_channel_info),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready to go.")
        print("\nNext steps:")
        print("1. Run the full pipeline: POST /full-pipeline")
        print("2. Check the API docs: http://localhost:8000/docs")
        print("3. Start creating YouTube Shorts!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all API keys are set in .env")
        print("2. Make sure the FastAPI server is running")
        print("3. Check the README.md for setup instructions")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 