#!/usr/bin/env python3
"""
Test script to verify image generator integration
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_available_generators():
    """Test the available image generators endpoint"""
    print("Testing available image generators...")
    
    try:
        response = requests.get(f"{BASE_URL}/available-image-generators")
        if response.status_code == 200:
            data = response.json()
            print("✅ Available generators:")
            for name, info in data["generators"].items():
                status = "✅ Enabled" if info["enabled"] else "❌ Disabled"
                print(f"  {name}: {status}")
                print(f"    Description: {info['description']}")
            return True
        else:
            print(f"❌ Failed to get generators: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing generators: {e}")
        return False

def test_dalle_generation():
    """Test DALL-E image generation"""
    print("\nTesting DALL-E image generation...")
    
    test_lines = [
        "A person looking confused and scratching their head",
        "Someone with a puzzled expression"
    ]
    
    payload = {
        "script_lines": test_lines,
        "style": "realistic",
        "use_stable_diffusion": False
    }
    
    try:
        response = requests.post(f"{BASE_URL}/generate-images", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ DALL-E generation successful: {data['message']}")
            print(f"   Generated {data['total_generated']}/{data['total_requested']} images")
            return True
        else:
            print(f"❌ DALL-E generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing DALL-E: {e}")
        return False

def test_stable_diffusion_generation():
    """Test Stable Diffusion image generation"""
    print("\nTesting Stable Diffusion image generation...")
    
    test_lines = [
        "A person looking confused and scratching their head",
        "Someone with a puzzled expression"
    ]
    
    payload = {
        "script_lines": test_lines,
        "style": "realistic",
        "use_stable_diffusion": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/generate-images", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stable Diffusion generation successful: {data['message']}")
            print(f"   Generated {data['total_generated']}/{data['total_requested']} images")
            return True
        else:
            print(f"❌ Stable Diffusion generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing Stable Diffusion: {e}")
        return False

def test_full_pipeline_with_dalle():
    """Test full pipeline with DALL-E"""
    print("\nTesting full pipeline with DALL-E...")
    
    payload = {
        "story_type": "comedy",
        "use_ffmpeg": False,
        "use_stable_diffusion": False
    }
    
    try:
        response = requests.post(f"{BASE_URL}/full-pipeline-with-images", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Full pipeline with DALL-E successful: {data['message']}")
            print(f"   Image generator: {data['image_generator']}")
            return True
        else:
            print(f"❌ Full pipeline with DALL-E failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing full pipeline with DALL-E: {e}")
        return False

def test_full_pipeline_with_stable_diffusion():
    """Test full pipeline with Stable Diffusion"""
    print("\nTesting full pipeline with Stable Diffusion...")
    
    payload = {
        "story_type": "comedy",
        "use_ffmpeg": False,
        "use_stable_diffusion": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/full-pipeline-with-images", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Full pipeline with Stable Diffusion successful: {data['message']}")
            print(f"   Image generator: {data['image_generator']}")
            return True
        else:
            print(f"❌ Full pipeline with Stable Diffusion failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing full pipeline with Stable Diffusion: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Image Generator Integration")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print("❌ API is not running. Please start the server first.")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Please make sure the server is running on http://localhost:8000")
        return
    
    print("✅ API is running")
    
    # Run tests
    tests = [
        test_available_generators,
        test_dalle_generation,
        test_stable_diffusion_generation,
        test_full_pipeline_with_dalle,
        test_full_pipeline_with_stable_diffusion
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            time.sleep(1)  # Small delay between tests
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Image generator integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n💡 Usage Tips:")
    print("- Use 'use_stable_diffusion: false' for DALL-E (default)")
    print("- Use 'use_stable_diffusion: true' for Stable Diffusion")
    print("- Check /available-image-generators for generator status")

if __name__ == "__main__":
    main() 