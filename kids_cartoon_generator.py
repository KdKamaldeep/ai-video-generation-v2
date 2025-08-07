#!/usr/bin/env python3
"""
Kids Cartoon Generator - YouTube Shorts

This generator creates kid-friendly cartoon content for YouTube Shorts using:
- AnimateDiff for cartoon-style video generation
- Script generation for kid-friendly stories
- Voice synthesis for narration
- FFmpeg for video composition

Usage: python kids_cartoon_generator.py
"""

import os
import sys
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KidsCartoonGenerator:
    def __init__(self):
        """Initialize the kids cartoon generator"""
        self.output_dir = "output/kids_cartoons"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Kid-friendly story types
        self.kids_story_types = {
            "educational": "fun and educational content for learning",
            "adventure": "exciting adventures with friendly characters",
            "animals": "cute animal stories and facts",
            "colors": "colorful and vibrant learning content",
            "numbers": "fun number learning and counting",
            "shapes": "geometric shapes and patterns",
            "nature": "nature exploration and environmental awareness",
            "friendship": "stories about friendship and kindness",
            "imagination": "creative and imaginative stories",
            "music": "musical and rhythmic content"
        }
        
        # Kid-friendly cartoon styles
        self.cartoon_styles = {
            "cute": "cute, adorable, child-friendly, soft colors, rounded shapes",
            "bright": "bright, vibrant, colorful, cheerful, energetic",
            "simple": "simple, clean, minimalist, easy to understand",
            "whimsical": "whimsical, magical, fantastical, dreamy",
            "educational": "clear, educational, informative, engaging"
        }
        
        logger.info("🎨 Kids Cartoon Generator initialized")
        logger.info(f"Available story types: {list(self.kids_story_types.keys())}")
        logger.info(f"Available cartoon styles: {list(self.cartoon_styles.keys())}")
    
    def generate_kids_cartoon(self, 
                             story_type: str = "educational",
                             cartoon_style: str = "cute",
                             duration_seconds: int = 60,
                             include_voice: bool = True) -> Dict[str, any]:
        """
        Generate a complete kids cartoon for YouTube Shorts
        
        Args:
            story_type: Type of story (educational, adventure, animals, etc.)
            cartoon_style: Visual style (cute, bright, simple, etc.)
            duration_seconds: Target duration in seconds
            include_voice: Whether to include voice narration
            
        Returns:
            Dictionary with paths to generated files
        """
        logger.info("🎬 Starting Kids Cartoon Generation")
        logger.info("=" * 60)
        
        results = {
            "success": False,
            "story_type": story_type,
            "cartoon_style": cartoon_style,
            "duration": duration_seconds,
            "files": {}
        }
        
        try:
            # Step 1: Generate kid-friendly script
            logger.info("📝 Step 1: Generating kid-friendly script...")
            script = self._generate_kids_script(story_type, duration_seconds)
            if not script:
                logger.error("❌ Failed to generate script")
                return results
            
            results["script"] = script
            logger.info(f"✅ Script generated: {script['title']}")
            
            # Step 2: Generate cartoon videos for each scene
            logger.info("🎨 Step 2: Generating cartoon videos...")
            video_paths = self._generate_cartoon_videos(script, cartoon_style)
            if not video_paths:
                logger.error("❌ Failed to generate videos")
                return results
            
            results["files"]["videos"] = video_paths
            logger.info(f"✅ Generated {len(video_paths)} cartoon videos")
            
            # Step 3: Generate voice narration
            audio_path = None
            if include_voice:
                logger.info("🎤 Step 3: Generating voice narration...")
                audio_path = self._generate_voice_narration(script)
                if audio_path:
                    results["files"]["audio"] = audio_path
                    logger.info(f"✅ Voice narration generated: {os.path.basename(audio_path)}")
                else:
                    logger.warning("⚠️ Voice generation failed, continuing without audio")
            
            # Step 4: Combine videos and audio
            logger.info("🎬 Step 4: Combining videos and audio...")
            final_video = self._combine_video_and_audio(video_paths, audio_path, script)
            if final_video:
                results["files"]["final_video"] = final_video
                logger.info(f"✅ Final cartoon created: {os.path.basename(final_video)}")
            else:
                logger.error("❌ Failed to combine video and audio")
                return results
            
            # Step 5: Upload to S3 (optional)
            logger.info("☁️ Step 5: Uploading to S3...")
            s3_url = self._upload_to_s3(final_video, story_type)
            if s3_url:
                results["files"]["s3_url"] = s3_url
                logger.info(f"✅ Uploaded to S3: {s3_url}")
            
            results["success"] = True
            logger.info("🎉 Kids Cartoon Generation Completed Successfully!")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Kids cartoon generation failed: {e}")
            return results
    
    def _generate_kids_script(self, story_type: str, duration_seconds: int) -> Optional[Dict]:
        """Generate a kid-friendly script"""
        try:
            from utils.script_generator import ScriptGenerator
            
            # Create a custom kids-friendly prompt
            kids_prompt = f"""
            Create a {story_type} story for kids aged 3-8 years old.
            The story should be:
            - Educational and fun
            - Age-appropriate content
            - Simple language
            - Engaging for children
            - Around {duration_seconds} seconds long
            - Include visual descriptions for cartoon scenes
            """
            
            script_generator = ScriptGenerator()
            
            # Generate script with custom prompt
            script = script_generator.generate_script(story_type)
            
            # Convert to our format
            return {
                "title": script.title,
                "narration": [line.text for line in script.narration],
                "visual_suggestions": [line.visual_suggestion for line in script.narration],
                "durations": [line.duration for line in script.narration],
                "total_duration": script.total_duration,
                "tags": script.tags
            }
            
        except ImportError as e:
            logger.warning(f"Script generator import failed: {e}")
            logger.info("Using fallback script")
            return self._generate_fallback_kids_script(story_type, duration_seconds)
        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            logger.info("Using fallback script")
            # Fallback script
            return self._generate_fallback_kids_script(story_type, duration_seconds)
    
    def _generate_fallback_kids_script(self, story_type: str, duration_seconds: int) -> Dict:
        """Generate a fallback kids script if API fails"""
        fallback_scripts = {
            "educational": {
                "title": "Learning Colors with Friends",
                "narration": [
                    "Hello little friends! Let's learn about colors today!",
                    "Look at this beautiful red apple. Red is the color of love!",
                    "Here's a bright yellow sun. Yellow makes us happy!",
                    "And this is a big blue sky. Blue is peaceful and calm!",
                    "Colors are everywhere around us. What's your favorite color?"
                ],
                "visual_suggestions": [
                    "Friendly cartoon character waving hello",
                    "Bright red apple with a happy face",
                    "Cheerful yellow sun with sunglasses",
                    "Peaceful blue sky with fluffy clouds",
                    "Rainbow with all the colors"
                ],
                "durations": [3, 4, 4, 4, 3],
                "total_duration": 18,
                "tags": ["educational", "colors", "kids", "learning"]
            },
            "animals": {
                "title": "Fun Animal Friends",
                "narration": [
                    "Meet our animal friends! They love to play together!",
                    "This is Sammy the squirrel. He loves to climb trees!",
                    "Here's Bella the bunny. She hops around all day!",
                    "And this is Tommy the turtle. He walks slowly but surely!",
                    "All animals are special in their own way!"
                ],
                "visual_suggestions": [
                    "Group of cute cartoon animals",
                    "Squirrel climbing a tree",
                    "Bunny hopping in a garden",
                    "Turtle walking slowly",
                    "Animals playing together"
                ],
                "durations": [3, 4, 4, 4, 3],
                "total_duration": 18,
                "tags": ["animals", "friendship", "kids", "nature"]
            }
        }
        
        return fallback_scripts.get(story_type, fallback_scripts["educational"])
    
    def _generate_cartoon_videos(self, script: Dict, cartoon_style: str) -> List[str]:
        """Generate cartoon videos for each scene"""
        try:
            from utils.animatediff_generator import AnimateDiffGenerator
            
            generator = AnimateDiffGenerator(
                sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
                motion_adapter_id="guoyww/animatediff-v1-5-2",
                memory_optimization=True,
                cache_dir="models_cache"  # Cache models locally
            )
            
            video_paths = []
            style_enhancement = self.cartoon_styles.get(cartoon_style, self.cartoon_styles["cute"])
            
            for i, (narration, visual_suggestion) in enumerate(zip(script["narration"], script["visual_suggestions"])):
                logger.info(f"  Generating scene {i+1}/{len(script['narration'])}...")
                
                # Create kid-friendly prompt
                prompt = f"{visual_suggestion}, {style_enhancement}, kid-friendly, safe for children, no scary elements"
                
                # Generate video with kid-appropriate settings
                video_path = generator.generate_animated_video_from_text(
                    text=prompt,
                    style="cartoon",  # Use cartoon style
                    width=512,
                    height=768,  # Vertical for YouTube Shorts
                    num_frames=16,
                    fps=8,
                    motion_strength=0.6,  # Gentle motion for kids
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    seed=42 + i,
                    decode_chunk_size=8
                )
                
                if video_path:
                    video_paths.append(video_path)
                    logger.info(f"    ✅ Scene {i+1} generated")
                else:
                    logger.error(f"    ❌ Failed to generate scene {i+1}")
            
            generator.cleanup()
            return video_paths
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return []
    
    def _generate_voice_narration(self, script: Dict) -> Optional[str]:
        """Generate voice narration for the script"""
        try:
            from utils.unified_voice_synthesizer import UnifiedVoiceSynthesizer
            
            # Create output path
            timestamp = int(time.time())
            audio_path = os.path.join(self.output_dir, f"kids_narration_{timestamp}.wav")
            
            # Initialize unified voice synthesizer
            # You can switch between "coqui" and "elevenlabs"
            voice_provider = os.getenv("VOICE_PROVIDER", "coqui")  # Default to Coqui TTS
            
            voice_config = {
                "gpu": True,  # Use GPU for Coqui TTS
                "speaker": "random",  # Random speaker for variety
                "voice_dir": "bark_voices/",
                "text_temp": 0.7,
                "waveform_temp": 0.7
            }
            
            synthesizer = UnifiedVoiceSynthesizer(
                provider=voice_provider,
                config=voice_config
            )
            
            # Generate voice for all narration lines
            audio_file = synthesizer.synthesize_voice(script["narration"], audio_path)
            
            # Cleanup
            synthesizer.cleanup()
            
            return audio_file if os.path.exists(audio_file) else None
            
        except Exception as e:
            logger.error(f"Voice generation failed: {e}")
            return None
    
    def _combine_video_and_audio(self, video_paths: List[str], audio_path: Optional[str], script: Dict) -> Optional[str]:
        """Combine videos and audio into final cartoon"""
        try:
            # Try to import ffmpeg
            try:
                import ffmpeg
                logger.info("Using ffmpeg-python for video processing")
            except ImportError:
                logger.warning("ffmpeg-python not available, trying alternative methods")
                return self._combine_video_fallback(video_paths, audio_path, script)
            
            # Create output path
            timestamp = int(time.time())
            final_video_path = os.path.join(self.output_dir, f"kids_cartoon_{timestamp}.mp4")
            
            # First, combine all video segments
            combined_video_path = os.path.join(self.output_dir, f"combined_video_{timestamp}.mp4")
            
            # Create file list for video concatenation
            file_list_path = os.path.join(self.output_dir, "video_list.txt")
            with open(file_list_path, 'w') as f:
                for video_path in video_paths:
                    f.write(f"file '{os.path.abspath(video_path)}'\n")
            
            # Combine videos
            (
                ffmpeg
                .input(file_list_path, f='concat', safe=0)
                .output(combined_video_path, c='copy')
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Clean up file list
            os.remove(file_list_path)
            
            # Add audio if available
            if audio_path and os.path.exists(audio_path):
                # Combine video with audio
                (
                    ffmpeg
                    .input(combined_video_path)
                    .input(audio_path)
                    .output(final_video_path, 
                           vcodec='copy',
                           acodec='aac',
                           shortest=None)  # Match audio duration
                    .overwrite_output()
                    .run(quiet=True)
                )
                
                # Clean up intermediate file
                os.remove(combined_video_path)
            else:
                # No audio, just rename the combined video
                os.rename(combined_video_path, final_video_path)
            
            return final_video_path if os.path.exists(final_video_path) else None
            
        except Exception as e:
            logger.error(f"Video combination failed: {e}")
            logger.info("Trying fallback method...")
            return self._combine_video_fallback(video_paths, audio_path, script)
    
    def _combine_video_fallback(self, video_paths: List[str], audio_path: Optional[str], script: Dict) -> Optional[str]:
        """Fallback method for video combination without ffmpeg-python"""
        try:
            logger.info("Using fallback video combination method")
            
            # Create output path
            timestamp = int(time.time())
            final_video_path = os.path.join(self.output_dir, f"kids_cartoon_fallback_{timestamp}.mp4")
            
            # Simple approach: just use the first video if available
            if video_paths and os.path.exists(video_paths[0]):
                import shutil
                shutil.copy2(video_paths[0], final_video_path)
                logger.info(f"Fallback: Using first video as final output")
                return final_video_path
            else:
                logger.error("No valid videos found for fallback")
                return None
                
        except Exception as e:
            logger.error(f"Fallback video combination failed: {e}")
            return None
    
    def _upload_to_s3(self, video_path: str, story_type: str) -> Optional[str]:
        """Upload the final cartoon to S3"""
        try:
            from utils.s3_uploader import S3Uploader
            
            uploader = S3Uploader(bucket_name="why-would-you")
            
            # Generate S3 key
            timestamp = int(time.time())
            filename = os.path.basename(video_path)
            s3_key = f"kids-cartoons/{story_type}/{timestamp}/{filename}"
            
            # Upload to S3
            s3_url = uploader.upload_video(
                local_file_path=video_path,
                s3_key=s3_key,
                folder="kids-cartoons"
            )
            
            return s3_url
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return None

def main():
    """Main function to generate kids cartoons"""
    logger.info("🎨 Kids Cartoon Generator for YouTube Shorts")
    logger.info("=" * 60)
    
    generator = KidsCartoonGenerator()
    
    # Example: Generate an educational cartoon
    logger.info("Creating educational cartoon for kids...")
    result = generator.generate_kids_cartoon(
        story_type="educational",
        cartoon_style="cute",
        duration_seconds=60,
        include_voice=True
    )
    
    if result["success"]:
        logger.info("🎉 Successfully generated kids cartoon!")
        logger.info(f"📁 Final video: {result['files'].get('final_video', 'N/A')}")
        logger.info(f"🌐 S3 URL: {result['files'].get('s3_url', 'N/A')}")
    else:
        logger.error("❌ Failed to generate kids cartoon")

if __name__ == "__main__":
    main() 