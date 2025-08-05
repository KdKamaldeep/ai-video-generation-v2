import os
import tempfile
from utils.shotstack_video_creator import ShotstackVideoCreator
from utils.script_generator import ScriptGenerator

def test_video_creation():
    """Test the video creation functionality"""
    print("Testing video creation...")
    
    # Initialize video creator
    video_creator = ShotstackVideoCreator()
    
    # Create a simple test script
    test_script = {
        "title": "Test Video",
        "narration": [
            {
                "text": "This is a test narration line.",
                "duration": 3.0,
                "visual_suggestion": "test image"
            },
            {
                "text": "This is another test line to see if video works.",
                "duration": 4.0,
                "visual_suggestion": "another test image"
            }
        ]
    }
    
    # Create a simple audio file (or use existing one)
    audio_path = "output/tmprxxxn3j9.mp3"  # Use existing audio file
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return
    
    # Create video
    output_path = "output/test_video.mp4"
    
    try:
        print(f"Creating video with audio: {audio_path}")
        result = video_creator.create_video(
            audio_path=audio_path,
            narration_lines=test_script["narration"],
            output_path=output_path
        )
        
        print(f"Video created successfully: {result}")
        
        # Check if file exists and has content
        if os.path.exists(result):
            file_size = os.path.getsize(result)
            print(f"Video file size: {file_size} bytes")
            
            if file_size > 1000:  # Should be at least 1KB
                print("✅ Video creation test PASSED!")
            else:
                print("❌ Video file is too small, may not have video content")
        else:
            print("❌ Video file was not created")
            
    except Exception as e:
        print(f"❌ Error creating video: {e}")

if __name__ == "__main__":
    test_video_creation() 