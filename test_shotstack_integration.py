import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_shotstack_api_key():
    """Test if Shotstack API key is configured"""
    print("🔍 Testing Shotstack API configuration...")
    
    api_key = os.getenv("SHOTSTACK_API_KEY")
    if not api_key or api_key == "your_shotstack_api_key_here":
        print("❌ SHOTSTACK_API_KEY not configured")
        print("Please add your Shotstack API key to the .env file")
        print("Get your API key from: https://shotstack.io/")
        return False
    else:
        print("✅ SHOTSTACK_API_KEY is configured")
        return True

def test_shotstack_import():
    """Test if ShotstackVideoCreator can be imported"""
    print("\n🔍 Testing ShotstackVideoCreator import...")
    
    try:
        from utils.shotstack_video_creator import ShotstackVideoCreator
        print("✅ ShotstackVideoCreator imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import ShotstackVideoCreator: {e}")
        return False
    except Exception as e:
        print(f"❌ Error initializing ShotstackVideoCreator: {e}")
        return False

def test_shotstack_initialization():
    """Test if ShotstackVideoCreator can be initialized"""
    print("\n🔍 Testing ShotstackVideoCreator initialization...")
    
    try:
        from utils.shotstack_video_creator import ShotstackVideoCreator
        creator = ShotstackVideoCreator()
        print("✅ ShotstackVideoCreator initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize ShotstackVideoCreator: {e}")
        return False

def test_shotstack_api_connection():
    """Test if we can connect to Shotstack API"""
    print("\n🔍 Testing Shotstack API connection...")
    
    try:
        from utils.shotstack_video_creator import ShotstackVideoCreator
        creator = ShotstackVideoCreator()
        
        # Try to list renders to test API connection
        renders = creator.list_renders(limit=1)
        print("✅ Shotstack API connection successful")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Shotstack API: {e}")
        print("This might be due to an invalid API key or network issues")
        return False

def main():
    """Run all Shotstack integration tests"""
    print("🚀 Shotstack Integration Test Suite")
    print("=" * 40)
    
    tests = [
        test_shotstack_api_key,
        test_shotstack_import,
        test_shotstack_initialization,
        test_shotstack_api_connection
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