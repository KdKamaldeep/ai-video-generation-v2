#!/usr/bin/env python3
"""
Demo script for Stable Video Diffusion integration with FFmpeg video creation

This script demonstrates how to:
1. Generate images using Stable Diffusion
2. Create motion videos from those images using Stable Video Diffusion
3. Combine everything into a final video using FFmpeg
"""

import os
import sys
import logging
from typing import List, Dict
import tempfile

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stable_diffusion_generator import StableDiffusionGenerator
from utils.ffmpeg_video_creator import FFmpegVideoCreator
from utils.voice_synthesizer import VoiceSynthesizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_single_image_to_video():
    """Demo: Convert a single image to a motion video"""
    logger.info("=== Demo: Single Image to Motion Video ===")
    
    try:
        # Initialize stable diffusion generator
        sd_generator = StableDiffusionGenerator()
        
        # Generate an image
        logger.info("Generating a sample image...")
        image_path = sd_generator.generate_image_from_text(
            text="A person walking through a beautiful garden with flowers blooming",
            style="realistic",
            width=1024,
            height=576  # 16:9 aspect ratio for video
        )
        
        if image_path:
            logger.info(f"Image generated: {image_path}")
            
            # Generate motion video from the image
            logger.info("Generating motion video from image...")
            video_path = sd_generator.generate_video_from_image(
                image_path=image_path,
                motion_strength=0.8,
                num_frames=25,
                fps=8,
                seed=42
            )
            
            if video_path:
                logger.info(f"Motion video generated: {video_path}")
                logger.info("Demo completed successfully!")
            else:
                logger.error("Failed to generate motion video")
        else:
            logger.error("Failed to generate image")
            
    except Exception as e:
        logger.error(f"Error in single image demo: {e}")
    finally:
        if 'sd_generator' in locals():
            sd_generator.cleanup()

def demo_script_to_motion_videos():
    """Demo: Generate motion videos for a script"""
    logger.info("=== Demo: Script to Motion Videos ===")
    
    # Sample script
    script_lines = [
        "A person walking through a beautiful garden",
        "They stop to admire the blooming flowers",
        "A butterfly lands on their hand",
        "They smile and continue their journey"
    ]
    
    try:
        # Initialize stable diffusion generator
        sd_generator = StableDiffusionGenerator()
        
        # Set up character consistency
        sd_generator.set_character_consistency(
            character_description="A young person with a peaceful expression",
            seed=12345
        )
        
        # Generate motion videos for each script line
        logger.info("Generating motion videos for script...")
        video_paths = sd_generator.generate_videos_for_script(
            script_lines=script_lines,
            style="realistic",
            motion_strength=0.7,
            num_frames=20,
            fps=8,
            maintain_character_consistency=True
        )
        
        # Check results
        successful_videos = [path for path in video_paths if path is not None]
        logger.info(f"Generated {len(successful_videos)}/{len(script_lines)} motion videos")
        
        for i, video_path in enumerate(video_paths):
            if video_path:
                logger.info(f"Video {i+1}: {video_path}")
            else:
                logger.warning(f"Video {i+1}: Failed to generate")
                
    except Exception as e:
        logger.error(f"Error in script demo: {e}")
    finally:
        if 'sd_generator' in locals():
            sd_generator.cleanup()

def demo_full_video_creation():
    """Demo: Full video creation with motion videos and audio"""
    logger.info("=== Demo: Full Video Creation with Motion ===")
    
    # Sample narration script
    narration_lines = [
        {
            "text": "Every day is a chance for a fresh start",
            "visual_suggestion": "A person watching the sunrise from their window",
            "duration": 3.0
        },
        {
            "text": "Take that first step towards your dreams",
            "visual_suggestion": "A person walking confidently down a path",
            "duration": 3.0
        },
        {
            "text": "You have the power to change your story",
            "visual_suggestion": "A person reaching up towards the sky",
            "duration": 3.0
        }
    ]
    
    try:
        # Initialize voice synthesizer
        voice_synth = VoiceSynthesizer()
        
        # Generate audio narration
        logger.info("Generating audio narration...")
        audio_path = voice_synth.create_narration_audio(
            text=" ".join([line["text"] for line in narration_lines]),
            output_path="output/demo_narration.mp3"
        )
        
        if not audio_path:
            logger.error("Failed to generate audio")
            return
        
        # Initialize video creator with stable video diffusion
        video_creator = FFmpegVideoCreator(use_stable_video_diffusion=True)
        
        # Create video with motion
        logger.info("Creating video with motion...")
        output_path = "output/demo_motion_video.mp4"
        
        video_path = video_creator.create_video_with_motion(
            audio_path=audio_path,
            narration_lines=narration_lines,
            output_path=output_path,
            motion_strength=0.8,
            num_frames_per_segment=25,
            video_fps=8
        )
        
        if video_path:
            logger.info(f"Full video created successfully: {video_path}")
            logger.info("Demo completed!")
        else:
            logger.error("Failed to create full video")
            
    except Exception as e:
        logger.error(f"Error in full video demo: {e}")
    finally:
        if 'video_creator' in locals():
            video_creator.cleanup()

def demo_image_sequence_to_video():
    """Demo: Convert a sequence of images to a motion video"""
    logger.info("=== Demo: Image Sequence to Motion Video ===")
    
    try:
        # Initialize stable diffusion generator
        sd_generator = StableDiffusionGenerator()
        
        # Generate a sequence of related images
        image_prompts = [
            "A person starting their morning routine",
            "The same person getting dressed and ready",
            "The person walking out the door with confidence",
            "The person arriving at their destination"
        ]
        
        logger.info("Generating image sequence...")
        image_paths = []
        
        for i, prompt in enumerate(image_prompts):
            image_path = sd_generator.generate_image_from_text(
                text=prompt,
                style="realistic",
                seed=1000 + i  # Consistent seed variation
            )
            if image_path:
                image_paths.append(image_path)
                logger.info(f"Generated image {i+1}: {image_path}")
        
        if len(image_paths) >= 2:
            # Create motion video from image sequence
            logger.info("Creating motion video from image sequence...")
            output_path = "output/demo_sequence_video.mp4"
            
            video_path = sd_generator.create_motion_video_from_image_sequence(
                image_paths=image_paths,
                output_path=output_path,
                motion_strength=0.6,
                num_frames_per_image=15,
                fps=8,
                transition_frames=3
            )
            
            if video_path:
                logger.info(f"Sequence video created: {video_path}")
            else:
                logger.error("Failed to create sequence video")
        else:
            logger.error("Not enough images generated for sequence")
            
    except Exception as e:
        logger.error(f"Error in sequence demo: {e}")
    finally:
        if 'sd_generator' in locals():
            sd_generator.cleanup()

def main():
    """Run all demos"""
    logger.info("Starting Stable Video Diffusion Demos")
    
    # Create output directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/videos", exist_ok=True)
    
    # Run demos
    try:
        # Demo 1: Single image to video
        demo_single_image_to_video()
        
        print("\n" + "="*50 + "\n")
        
        # Demo 2: Script to motion videos
        demo_script_to_motion_videos()
        
        print("\n" + "="*50 + "\n")
        
        # Demo 3: Image sequence to video
        demo_image_sequence_to_video()
        
        print("\n" + "="*50 + "\n")
        
        # Demo 4: Full video creation (requires voice synthesis)
        demo_full_video_creation()
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    logger.info("All demos completed!")

if __name__ == "__main__":
    main() 