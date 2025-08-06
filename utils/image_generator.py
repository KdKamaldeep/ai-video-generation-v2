import os
import requests
import logging
import openai
import tempfile
import uuid
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Create output directory for images
        self.output_dir = "output/images"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("ImageGenerator initialized successfully")
    
    def generate_image_from_text(self, text: str, style: str = "realistic") -> Optional[str]:
        """
        Generate an image from text using DALL-E and save locally
        
        Args:
            text: Text description for image generation
            style: Image style (realistic, cartoon, minimalist, etc.)
            
        Returns:
            Local path to the generated image, or None if failed
        """
        try:
            # Create enhanced prompt for better image generation
            enhanced_prompt = self._create_enhanced_prompt(text, style)
            
            logger.info(f"Generating image for text: {text[:50]}...")
            
            # Generate image using DALL-E
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size="1024x1792",
                quality="standard",
                n=1,
            )
            
            # Get image URL
            image_url = response.data[0].url
            
            # Download and save locally
            local_path = self._download_and_save_image(image_url, text)
            
            if local_path:
                logger.info(f"Image generated and saved locally: {local_path}")
                return local_path
            else:
                logger.error("Failed to save image locally")
                return None
                
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None
    
    def generate_images_for_script(self, script_lines: List[str], style: str = "realistic") -> List[Optional[str]]:
        """
        Generate images for each script line and save locally
        
        Args:
            script_lines: List of script text lines
            style: Image style for generation
            
        Returns:
            List of local paths for generated images
        """
        image_paths = []
        
        for i, line in enumerate(script_lines):
            logger.info(f"Generating image {i+1}/{len(script_lines)} for: {line[:50]}...")
            
            # Generate image for this line
            image_path = self.generate_image_from_text(line, style)
            image_paths.append(image_path)
            
            # Add small delay to avoid rate limiting
            import time
            time.sleep(1)
        
        return image_paths
    
    def _create_enhanced_prompt(self, text: str, style: str) -> str:
        style_prompts = {
            "realistic": "high-quality, realistic, detailed, professional photography",
            "cartoon": "colorful, cartoon-style, animated, fun, vibrant",
            "minimalist": "minimalist, clean, simple, modern design",
            "dramatic": "dramatic lighting, cinematic, moody, atmospheric",
            "funny": "humorous, comedic, lighthearted, playful",
            "relatable": "everyday life, relatable, authentic, candid"
        }
        
        style_desc = style_prompts.get(style, style_prompts["realistic"])
        
        # Clean and enhance the text
        clean_text = text.strip().replace('"', '').replace("'", "")
        
        # Check if the text contains motion cues for video generation
        motion_keywords = [
            "camera", "panning", "dollying", "tilting", "zooming", "tracking", "creeping",
            "walking", "running", "falling", "rising", "spinning", "swaying", "moving",
            "blowing", "flowing", "falling", "rising", "breathing", "blinking", "fidgeting",
            "trembling", "twitching", "nodding", "vibration", "flickering", "shadows"
        ]
        
        has_motion = any(keyword in clean_text.lower() for keyword in motion_keywords)
        
        if has_motion:
            # For motion cues, create a static frame that suggests the motion
            enhanced_prompt = (
                f"{clean_text}. "
                f"Style: {style_desc}. "
                f"High resolution, well-lit, clear composition. "
                f"Capture the moment that suggests the described motion. "
                f"Centered subject, 9:16 aspect ratio, cinematic framing."
            )
        else:
            # For static images, use the original approach
            enhanced_prompt = (
                f"{clean_text}. "
                f"Style: {style_desc}. "
                f"High resolution, well-lit, clear composition. "
                f"centered subject, 9:16 aspect ratio."
            )
        
        return enhanced_prompt
    
    def _download_and_save_image(self, image_url: str, text: str) -> Optional[str]:
        """
        Download image from URL and save locally
        
        Args:
            image_url: URL of the generated image
            text: Original text for filename
            
        Returns:
            Local path of saved image, or None if failed
        """
        try:
            # Download image
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_text = safe_text.replace(' ', '_')
            filename = f"{timestamp}_{unique_id}_{safe_text}.png"
            
            # Save to output directory
            output_path = os.path.join(self.output_dir, filename)
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Image saved to: {output_path}")
            return output_path
                
        except Exception as e:
            logger.error(f"Error downloading and saving image: {e}")
            return None
    
    def generate_thumbnail_for_video(self, title: str, style: str = "realistic") -> Optional[str]:
        """
        Generate a thumbnail image for the video
        
        Args:
            title: Video title
            style: Image style
            
        Returns:
            Local path to the thumbnail image
        """
        thumbnail_prompt = f"Thumbnail for YouTube video: {title}. Eye-catching, clickable thumbnail design."
        return self.generate_image_from_text(thumbnail_prompt, style) 