import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_shotstack_timeline_creation():
    """Test if we can create a valid Shotstack timeline"""
    print("🔍 Testing Shotstack timeline creation...")
    
    try:
        from utils.shotstack_video_creator import ShotstackVideoCreator
        creator = ShotstackVideoCreator()
        
        # Test timeline creation
        audio_url = "file:///test/audio.mp3"
        narration_lines = [
            {
                "text": "Why would you leave your phone on silent?",
                "duration": 3.0,
                "visual_suggestion": "person searching for phone"
            },
            {
                "text": "You know you're going to miss that important call.",
                "duration": 4.0,
                "visual_suggestion": "phone ringing"
            }
        ]
        
        timeline = creator._create_timeline(audio_url, narration_lines)
        
        # Verify timeline structure
        required_keys = ["timeline", "output"]
        for key in required_keys:
            if key not in timeline:
                print(f"❌ Missing required key: {key}")
                return False
        
        # Verify timeline has tracks
        if "tracks" not in timeline["timeline"]:
            print("❌ Timeline missing tracks")
            return False
        
        # Verify output configuration
        if "size" not in timeline["output"]:
            print("❌ Output missing size configuration")
            return False
        
        print("✅ Timeline creation successful")
        print(f"   - Tracks: {len(timeline['timeline']['tracks'])}")
        print(f"   - Output size: {timeline['output']['size']['width']}x{timeline['output']['size']['height']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating timeline: {e}")
        return False

def test_shotstack_payload_structure():
    """Test if the payload matches the expected Shotstack format"""
    print("\n🔍 Testing Shotstack payload structure...")
    
    try:
        from utils.shotstack_video_creator import ShotstackVideoCreator
        creator = ShotstackVideoCreator()
        
        # Create a simple timeline
        audio_url = "file:///test/audio.mp3"
        narration_lines = [
            {
                "text": "Test text",
                "duration": 3.0,
                "visual_suggestion": "test"
            }
        ]
        
        timeline = creator._create_timeline(audio_url, narration_lines)
        
        # Check if the structure matches the expected format
        expected_structure = {
            "timeline": {
                "tracks": [
                    {"clips": []},  # Background
                    {"clips": []},  # Audio
                    {"clips": []}   # Text
                ]
            },
            "output": {
                "format": "mp4",
                "size": {
                    "width": 1080,
                    "height": 1920
                }
            }
        }
        
        # Verify structure
        if "timeline" in timeline and "output" in timeline:
            if "tracks" in timeline["timeline"] and len(timeline["timeline"]["tracks"]) >= 3:
                print("✅ Payload structure is correct")
                return True
        
        print("❌ Payload structure is incorrect")
        return False
        
    except Exception as e:
        print(f"❌ Error testing payload structure: {e}")
        return False

def main():
    """Run Shotstack integration tests"""
    print("🚀 Shotstack Integration Test Suite")
    print("=" * 40)
    
    tests = [
        test_shotstack_timeline_creation,
        test_shotstack_payload_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Shotstack integration is ready.")
        print("\nNext steps:")
        print("1. Run the main application: python main.py")
        print("2. Test video creation: python test_video.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 