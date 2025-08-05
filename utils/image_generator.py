import os
import requests
import logging
import openai
from typing import List, Optional
from dotenv import load_dotenv

# Import S3 uploader
from .s3_uploader import S3Uploader

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
        
        # Initialize S3 uploader
        try:
            self.s3_uploader = S3Uploader()
            logger.info("ImageGenerator initialized with S3 uploader")
        except Exception as e:
            logger.error(f"Failed to initialize S3 uploader: {e}")
            self.s3_uploader = None
    
    def generate_image_from_text(self, text: str, style: str = "realistic") -> Optional[str]:
        """
        Generate an image from text using DALL-E and upload to S3
        
        Args:
            text: Text description for image generation
            style: Image style (realistic, cartoon, minimalist, etc.)
            
        Returns:
            S3 URL of the generated image, or None if failed
        """
        try:
            # Create enhanced prompt for better image generation
            enhanced_prompt = self._create_enhanced_prompt(text, style)
            
            logger.info(f"Generating image for text: {text[:50]}...")
            
            # Generate image using DALL-E
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            # Get image URL
            image_url = response.data[0].url
            
            # Download and upload to S3
            s3_url = self._download_and_upload_to_s3(image_url, text)
            
            if s3_url:
                logger.info(f"Image generated and uploaded to S3: {s3_url}")
                return s3_url
            else:
                logger.error("Failed to upload image to S3")
                return None
                
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None
    
    def generate_images_for_script(self, script_lines: List[str], style: str = "realistic") -> List[Optional[str]]:
        """
        Generate images for each script line and upload to S3
        
        Args:
            script_lines: List of script text lines
            style: Image style for generation
            
        Returns:
            List of S3 URLs for generated images
        """
        image_urls = []
        
        for i, line in enumerate(script_lines):
            logger.info(f"Generating image {i+1}/{len(script_lines)} for: {line[:50]}...")
            
            # Generate image for this line
            image_url = self.generate_image_from_text(line, style)
            image_urls.append(image_url)
            
            # Add small delay to avoid rate limiting
            import time
            time.sleep(1)
        
        return image_urls
    
    def _create_enhanced_prompt(self, text: str, style: str) -> str:
        """
        Create an enhanced prompt for better image generation
        """
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
        
        enhanced_prompt = f"{clean_text}. Style: {style_desc}. High resolution, well-lit, clear composition."
        
        return enhanced_prompt
    
    def _download_and_upload_to_s3(self, image_url: str, text: str) -> Optional[str]:
        """
        Download image from URL and upload to S3
        
        Args:
            image_url: URL of the generated image
            text: Original text for filename
            
        Returns:
            S3 URL of uploaded image, or None if failed
        """
        try:
            import tempfile
            import uuid
            from datetime import datetime
            
            # Download image
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_file.write(response.content)
                temp_path = tmp_file.name
            
            # Generate S3 key
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_text = safe_text.replace(' ', '_')
            s3_key = f"thumbs/{timestamp}_{unique_id}_{safe_text}.png"
            
            # Upload to S3
            if self.s3_uploader:
                s3_url = self.s3_uploader.upload_file_with_custom_key(
                    temp_path, 
                    s3_key, 
                    "image/png"
                )
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                return s3_url
            else:
                logger.error("S3 uploader not available")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading and uploading image: {e}")
            return None
    
    def generate_thumbnail_for_video(self, title: str, style: str = "realistic") -> Optional[str]:
        """
        Generate a thumbnail image for the video
        
        Args:
            title: Video title
            style: Image style
            
        Returns:
            S3 URL of the thumbnail image
        """
        thumbnail_prompt = f"Thumbnail for YouTube video: {title}. Eye-catching, clickable thumbnail design."
        return self.generate_image_from_text(thumbnail_prompt, style) 