import os
import logging
import torch
import tempfile
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import numpy as np
from PIL import Image
import json
from dotenv import load_dotenv

# AnimateDiff imports
try:
    from diffusers import AnimateDiffPipeline, DDIMScheduler
    ANIMATEDIFF_AVAILABLE = True
except ImportError:
    ANIMATEDIFF_AVAILABLE = False
    logging.warning("AnimateDiff not available. Install with: pip install diffusers[animatediff]")

load_dotenv()
logger = logging.getLogger(__name__)

class AnimateDiffGenerator:
    def __init__(self, 
                 sd_model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE",
                 animatediff_model_id: str = "guoyww/animatediff",
                 device: str = "auto"):
        """
        Initialize AnimateDiff generator with Stable Diffusion for text-to-image
        
        Args:
            sd_model_id: Hugging Face model ID for Stable Diffusion
            animatediff_model_id: Hugging Face model ID for AnimateDiff
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.sd_model_id = sd_model_id
        self.animatediff_model_id = animatediff_model_id
        self.device = self._get_device(device)
        
        # Character consistency settings
        self.character_seed = None
        self.character_embeddings = {}
        self.character_style = {}
        
        # Create output directories
        self.output_dir = "output/images"
        self.video_output_dir = "output/videos"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)
        
        # Initialize pipelines
        self.sd_pipeline = None
        self.animatediff_pipeline = None
        self._load_pipelines()
        
        logger.info(f"AnimateDiffGenerator initialized with SD model: {sd_model_id}")
        logger.info(f"AnimateDiff model: {animatediff_model_id}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_pipelines(self):
        """Load both Stable Diffusion and AnimateDiff pipelines"""
        try:
            # Load Stable Diffusion pipeline for text-to-image
            logger.info(f"Loading Stable Diffusion pipeline on {self.device}...")
            
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                self.sd_model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Configure scheduler for better quality
            try:
                scheduler = DPMSolverMultistepScheduler.from_pretrained(
                    self.sd_model_id,
                    subfolder="scheduler"
                )
                self.sd_pipeline.scheduler = scheduler
                logger.info("DPMSolverMultistepScheduler configured successfully")
            except Exception as e:
                logger.warning(f"Could not configure scheduler: {e}")
            
            # Move to device
            self.sd_pipeline = self.sd_pipeline.to(self.device)
            
            # Enable memory optimizations
            if hasattr(self.sd_pipeline, "enable_attention_slicing"):
                self.sd_pipeline.enable_attention_slicing()
            
            if hasattr(self.sd_pipeline, "enable_vae_slicing"):
                self.sd_pipeline.enable_vae_slicing()
            
            logger.info("Stable Diffusion pipeline loaded successfully")
            
            # Load AnimateDiff pipeline if available
            if ANIMATEDIFF_AVAILABLE:
                logger.info(f"Loading AnimateDiff pipeline on {self.device}...")
                
                self.animatediff_pipeline = AnimateDiffPipeline.from_pretrained(
                    self.animatediff_model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    variant="fp16" if self.device == "cuda" else None
                )
                
                # Configure scheduler for AnimateDiff
                try:
                    scheduler = DDIMScheduler.from_pretrained(
                        self.animatediff_model_id,
                        subfolder="scheduler"
                    )
                    self.animatediff_pipeline.scheduler = scheduler
                    logger.info("DDIMScheduler configured for AnimateDiff")
                except Exception as e:
                    logger.warning(f"Could not configure AnimateDiff scheduler: {e}")
                
                # Move to device
                self.animatediff_pipeline = self.animatediff_pipeline.to(self.device)
                
                # Enable memory optimizations
                if hasattr(self.animatediff_pipeline, "enable_attention_slicing"):
                    self.animatediff_pipeline.enable_attention_slicing()
                
                if hasattr(self.animatediff_pipeline, "enable_vae_slicing"):
                    self.animatediff_pipeline.enable_vae_slicing()
                
                logger.info("AnimateDiff pipeline loaded successfully")
            else:
                logger.warning("AnimateDiff not available - motion generation will be disabled")
                
        except Exception as e:
            logger.error(f"Error loading pipelines: {e}")
            raise
    
    def set_character_consistency(self, character_description: str, seed: Optional[int] = None):
        """
        Set up character consistency for future generations
        
        Args:
            character_description: Description of the character
            seed: Random seed for consistency (if None, will generate one)
        """
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        self.character_seed = seed
        self.character_style = {
            "description": character_description,
            "seed": seed,
            "prompt_enhancement": f"same character: {character_description}, consistent appearance, same person"
        }
        
        logger.info(f"Character consistency set with seed: {seed}")
    
    def generate_image_from_text(
        self, 
        text: str, 
        style: str = "realistic",
        width: int = 512,
        height: int = 768,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        negative_prompt: str = None,
        seed: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate an image from text using Stable Diffusion
        
        Args:
            text: Text description for image generation
            style: Image style (realistic, cinematic, artistic, etc.)
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            negative_prompt: What to avoid in the image
            seed: Random seed for reproducibility
            
        Returns:
            Local path to the generated image, or None if failed
        """
        try:
            # Use character seed if available
            if self.character_seed is not None and seed is None:
                seed = self.character_seed
            
            # Set random seed
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Create enhanced prompt
            enhanced_prompt = self._create_enhanced_prompt(text, style)
            
            # Set default negative prompt
            if negative_prompt is None:
                negative_prompt = self._get_default_negative_prompt()
            
            logger.info(f"Generating image for: {text[:50]}...")
            
            # Generate image
            result = self.sd_pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
            )
            
            # Get the image
            image = result.images[0]
            
            # Save image
            local_path = self._save_image(image, text, seed)
            
            if local_path:
                logger.info(f"Image generated and saved: {local_path}")
                return local_path
            else:
                logger.error("Failed to save image")
                return None
                
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None
    
    def generate_animated_video_from_text(
        self,
        text: str,
        style: str = "realistic",
        width: int = 512,
        height: int = 768,
        num_frames: int = 16,
        fps: int = 8,
        motion_strength: float = 0.8,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        negative_prompt: str = None,
        seed: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate an animated video directly from text using AnimateDiff
        
        Args:
            text: Text description for video generation
            style: Video style
            width: Video width
            height: Video height
            num_frames: Number of frames to generate
            fps: Frames per second
            motion_strength: Strength of motion (0.0 to 1.0)
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            negative_prompt: What to avoid in the video
            seed: Random seed for reproducibility
            
        Returns:
            Local path to the generated video, or None if failed
        """
        if not ANIMATEDIFF_AVAILABLE or self.animatediff_pipeline is None:
            logger.error("AnimateDiff not available. Cannot generate animated video.")
            return None
        
        try:
            # Use character seed if available
            if self.character_seed is not None and seed is None:
                seed = self.character_seed
            
            # Set random seed
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Create enhanced prompt
            enhanced_prompt = self._create_enhanced_prompt(text, style)
            
            # Set default negative prompt
            if negative_prompt is None:
                negative_prompt = self._get_default_negative_prompt()
            
            logger.info(f"Generating animated video for: {text[:50]}...")
            
            # Generate animated video
            result = self.animatediff_pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
            )
            
            # Get the video frames
            video_frames = result.frames[0]
            
            # Save video
            video_path = self._save_video(video_frames, text, seed, fps)
            
            if video_path:
                logger.info(f"Animated video generated and saved: {video_path}")
                return video_path
            else:
                logger.error("Failed to save animated video")
                return None
                
        except Exception as e:
            logger.error(f"Error generating animated video: {e}")
            return None
    
    def add_motion_to_image(
        self,
        image_path: str,
        motion_type: str = "subtle",
        num_frames: int = 16,
        fps: int = 8,
        motion_strength: float = 0.8,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Optional[str]:
        """
        Add motion to a static image using AnimateDiff
        
        Args:
            image_path: Path to the input image
            motion_type: Type of motion to add ("subtle", "dynamic", "camera_movement", etc.)
            num_frames: Number of frames to generate
            fps: Frames per second
            motion_strength: Strength of motion (0.0 to 1.0)
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the original image
            seed: Random seed for reproducibility
            
        Returns:
            Local path to the generated video, or None if failed
        """
        if not ANIMATEDIFF_AVAILABLE or self.animatediff_pipeline is None:
            logger.error("AnimateDiff not available. Cannot add motion to image.")
            return None
        
        try:
            logger.info(f"Adding motion to image: {image_path}")
            
            # Check if image file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return None
            
            # Set random seed
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Ensure image is in the right format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image for video generation
            target_size = (512, 768)  # Standard size for AnimateDiff
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            logger.info(f"Processing image with size: {image.size} for motion generation")
            
            # Create motion prompt based on motion type
            motion_prompt = self._create_motion_prompt(motion_type, motion_strength)
            
            # Generate motion video
            result = self.animatediff_pipeline(
                prompt=motion_prompt,
                image=image,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
            )
            
            # Get the video frames
            video_frames = result.frames[0]
            
            logger.info(f"Generated {len(video_frames)} video frames")
            
            # Save video
            video_path = self._save_video(video_frames, image_path, seed, fps)
            
            if video_path:
                logger.info(f"Motion video generated and saved: {video_path}")
                return video_path
            else:
                logger.error("Failed to save motion video")
                return None
                
        except Exception as e:
            logger.error(f"Error adding motion to image: {e}")
            return None
    
    def generate_images_for_script(
        self, 
        script_lines: List[str], 
        style: str = "realistic",
        maintain_character_consistency: bool = True
    ) -> List[Optional[str]]:
        """
        Generate images for each script line with optional character consistency
        
        Args:
            script_lines: List of script text lines
            style: Image style for generation
            maintain_character_consistency: Whether to maintain character consistency
            
        Returns:
            List of local paths for generated images
        """
        image_paths = []
        
        for i, line in enumerate(script_lines):
            logger.info(f"Generating image {i+1}/{len(script_lines)} for: {line[:50]}...")
            
            # Use character seed for consistency if enabled
            seed = None
            if maintain_character_consistency and self.character_seed is not None:
                seed = self.character_seed + i
            
            # Generate image for this line
            image_path = self.generate_image_from_text(
                text=line,
                style=style,
                seed=seed
            )
            image_paths.append(image_path)
            
            # Small delay to prevent memory issues
            import time
            time.sleep(0.5)
        
        return image_paths
    
    def generate_motion_videos_for_script(
        self,
        script_lines: List[str],
        style: str = "realistic",
        motion_type: str = "subtle",
        num_frames: int = 16,
        fps: int = 8,
        maintain_character_consistency: bool = True
    ) -> List[Optional[str]]:
        """
        Generate motion videos for each script line using AnimateDiff
        
        Args:
            script_lines: List of script text lines
            style: Video style for generation
            motion_type: Type of motion to add
            num_frames: Number of frames per video
            fps: Frames per second
            maintain_character_consistency: Whether to maintain character consistency
            
        Returns:
            List of local paths for generated videos
        """
        video_paths = []
        
        for i, line in enumerate(script_lines):
            logger.info(f"Generating motion video {i+1}/{len(script_lines)} for: {line[:50]}...")
            
            # Use character seed for consistency if enabled
            seed = None
            if maintain_character_consistency and self.character_seed is not None:
                seed = self.character_seed + i
            
            # Generate animated video directly from text
            video_path = self.generate_animated_video_from_text(
                text=line,
                style=style,
                num_frames=num_frames,
                fps=fps,
                seed=seed
            )
            video_paths.append(video_path)
            
            # Small delay to prevent memory issues
            import time
            time.sleep(1.0)
        
        return video_paths
    
    def _create_enhanced_prompt(self, text: str, style: str) -> str:
        """Create an enhanced prompt for better image generation"""
        style_prompts = {
            "realistic": "high quality, realistic, detailed, professional photography",
            "cinematic": "cinematic lighting, dramatic, professional cinematography",
            "artistic": "artistic, creative, beautiful composition",
            "cartoon": "cartoon style, animated, colorful, fun",
            "minimalist": "minimalist, clean, simple, modern",
            "dramatic": "dramatic lighting, moody, atmospheric",
            "funny": "humorous, comedic, lighthearted, playful",
            "relatable": "everyday life, relatable, authentic, natural"
        }
        
        style_desc = style_prompts.get(style, style_prompts["realistic"])
        
        # Clean the text
        clean_text = text.strip().replace('"', '').replace("'", "")
        
        # Add character consistency if available
        character_enhancement = ""
        if self.character_style:
            character_enhancement = f", {self.character_style['prompt_enhancement']}"
        
        # Enhanced prompt
        enhanced_prompt = f"{clean_text}, {style_desc}, high quality{character_enhancement}"
        
        return enhanced_prompt
    
    def _create_motion_prompt(self, motion_type: str, motion_strength: float) -> str:
        """Create a motion prompt based on the desired motion type"""
        motion_prompts = {
            "subtle": "gentle movement, slight motion, soft animation",
            "dynamic": "dynamic movement, energetic motion, lively animation",
            "camera_movement": "camera pan, camera movement, cinematic motion",
            "object_motion": "object movement, things moving, dynamic objects",
            "zoom": "zoom effect, camera zoom, close-up motion",
            "pan": "panning motion, horizontal movement, camera pan",
            "tilt": "tilting motion, vertical movement, camera tilt",
            "rotation": "rotating motion, spinning effect, circular movement"
        }
        
        base_motion = motion_prompts.get(motion_type, motion_prompts["subtle"])
        
        # Adjust motion strength in the prompt
        if motion_strength > 0.8:
            intensity = "intense, strong"
        elif motion_strength > 0.5:
            intensity = "moderate, balanced"
        else:
            intensity = "gentle, subtle"
        
        return f"{base_motion}, {intensity} motion, smooth animation"
    
    def _get_default_negative_prompt(self) -> str:
        """Get default negative prompt to avoid common issues"""
        return (
            "blurry, low quality, watermark, signature, text, logo, "
            "distorted, deformed, ugly, bad anatomy"
        )
    
    def _save_image(self, image: Image.Image, text: str, seed: Optional[int] = None) -> Optional[str]:
        """Save the generated image to disk"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_text = safe_text.replace(' ', '_')
            
            seed_suffix = f"_seed{seed}" if seed else ""
            filename = f"{timestamp}_{unique_id}_{safe_text}{seed_suffix}.png"
            
            # Save to output directory
            output_path = os.path.join(self.output_dir, filename)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(output_path, 'PNG', quality=95)
            
            logger.info(f"Image saved to: {output_path}")
            return output_path
                
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None
    
    def _save_video(self, video_frames: List, original_text: str, 
                   seed: Optional[int] = None, fps: int = 8) -> Optional[str]:
        """Save the generated video frames to disk"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            
            # Extract description from original text
            safe_text = "".join(c for c in original_text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_text = safe_text.replace(' ', '_')
            
            seed_suffix = f"_seed{seed}" if seed else ""
            filename = f"{timestamp}_{unique_id}_{safe_text}{seed_suffix}.mp4"
            
            # Save to video output directory
            output_path = os.path.join(self.video_output_dir, filename)
            
            # Ensure frames are in the correct format for export_to_video
            processed_frames = []
            for frame in video_frames:
                if hasattr(frame, 'size'):  # PIL Image object
                    import numpy as np
                    frame_array = np.array(frame)
                    processed_frames.append(frame_array)
                else:  # Already numpy array
                    processed_frames.append(frame)
            
            # Convert frames to video
            export_to_video(processed_frames, output_path, fps=fps)
            
            logger.info(f"Video saved to: {output_path}")
            return output_path
                
        except Exception as e:
            logger.error(f"Error saving video: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.sd_pipeline is not None:
            del self.sd_pipeline
            logger.info("Stable Diffusion pipeline cleaned up")
        
        if self.animatediff_pipeline is not None:
            del self.animatediff_pipeline
            logger.info("AnimateDiff pipeline cleaned up")
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None 