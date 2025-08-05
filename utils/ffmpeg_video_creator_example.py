"""
Example usage of FFmpegVideoCreator class.

This example demonstrates how to create a vertical YouTube Shorts video
using the FFmpegVideoCreator class.
"""

import os
from ffmpeg_video_creator import FFmpegVideoCreator

def example_usage():
    """Example of how to use the FFmpegVideoCreator class."""
    
    # Initialize the video creator
    creator = FFmpegVideoCreator()
    
    # Example data - replace with your actual data
    audio_path = "./output/audio.mp3"  # Local audio file
    
    narration_lines = [
        {
            "text": "Welcome to our amazing video!",
            "duration": 3.0,
            "visual_suggestion": "./output/images/image1.jpg"
        },
        {
            "text": "This is the second scene with different content",
            "duration": 4.0,
            "visual_suggestion": "./output/images/image2.jpg"
        },
        {
            "text": "And here's the final scene to wrap it up",
            "duration": 3.0,
            "visual_suggestion": "./output/images/image3.jpg"
        }
    ]
    
    output_path = "./output/final_video.mp4"
    
    try:
        # Create the video
        result_path = creator.create_video_with_images(
            audio_path=audio_path,
            narration_lines=narration_lines,
            output_path=output_path
        )
        
        print(f"Video created successfully: {result_path}")
        
    except Exception as e:
        print(f"Error creating video: {e}")

if __name__ == "__main__":
    example_usage() 