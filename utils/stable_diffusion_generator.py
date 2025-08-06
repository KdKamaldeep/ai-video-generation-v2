import os
import logging
import torch
import tempfile
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
# Add stable video diffusion imports
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import json
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class StableDiffusionGenerator:
    def __init__(self, model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE", device: str = "auto"):
        """
        Initialize Stable Diffusion generator
        
        Args:
            model_id: Hugging Face model ID for Stable Diffusion
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.model_id = model_id
        self.device = self._get_device(device)
        
        # Character consistency settings
        self.character_seed = None
        self.character_embeddings = {}
        self.character_style = {}
        
        # Create output directory
        self.output_dir = "output/images"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create video output directory
        self.video_output_dir = "output/videos"
        os.makedirs(self.video_output_dir, exist_ok=True)
        
        # Initialize pipelines
        self.pipeline = None
        self.video_pipeline = None
        self._load_pipeline()
        self._load_video_pipeline()
        
        logger.info(f"StableDiffusionGenerator initialized with model: {model_id}")
    
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
    
    def _load_pipeline(self):
        """Load the Stable Diffusion pipeline"""
        try:
            logger.info(f"Loading Stable Diffusion pipeline on {self.device}...")
            
            # Load pipeline with memory optimization
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # Disable safety checker for faster generation
                requires_safety_checker=False
            )
            
            # Use DPM++ 2M scheduler for better quality
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
            
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
            
            logger.info("Pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            raise
    
    def _load_video_pipeline(self):
        """Load the Stable Video Diffusion pipeline"""
        try:
            logger.info(f"Loading Stable Video Diffusion pipeline on {self.device}...")
            
            # Load stable video diffusion pipeline
            self.video_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None
            )
            
            # Move to device
            self.video_pipeline = self.video_pipeline.to(self.device)
            
            # Enable memory optimizations
            if hasattr(self.video_pipeline, "enable_attention_slicing"):
                self.video_pipeline.enable_attention_slicing()
            
            if hasattr(self.video_pipeline, "enable_vae_slicing"):
                self.video_pipeline.enable_vae_slicing()
            
            logger.info("Video pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading video pipeline: {e}")
            # Don't raise here, as video generation is optional
            self.video_pipeline = None
    
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
            result = self.pipeline(
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
                # Use character seed as base, add small variation for each frame
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
    
    def generate_consistent_character_frames(
        self,
        script_lines: List[str],
        character_description: str,
        style: str = "realistic",
        num_variations: int = 1
    ) -> List[List[Optional[str]]]:
        """
        Generate multiple variations of consistent character frames
        
        Args:
            script_lines: List of script text lines
            character_description: Description of the character
            style: Image style
            num_variations: Number of variations to generate
            
        Returns:
            List of lists, where each inner list contains paths for one variation
        """
        # Set up character consistency
        self.set_character_consistency(character_description)
        
        all_variations = []
        
        for variation in range(num_variations):
            logger.info(f"Generating variation {variation + 1}/{num_variations}")
            
            # Generate base seed for this variation
            base_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
            variation_paths = []
            for i, line in enumerate(script_lines):
                # Use base seed + frame index for consistency within variation
                frame_seed = base_seed + i
                
                image_path = self.generate_image_from_text(
                    text=line,
                    style=style,
                    seed=frame_seed
                )
                variation_paths.append(image_path)
                
                import time
                time.sleep(0.5)
            
            all_variations.append(variation_paths)
        
        return all_variations
    
    def _create_enhanced_prompt(self, text: str, style: str) -> str:
        """Create an enhanced prompt for better image generation"""
        style_prompts = {
            "realistic": "high quality, realistic, detailed, professional photography, sharp focus",
            "cinematic": "cinematic lighting, movie still, dramatic, professional cinematography",
            "artistic": "artistic, creative, beautiful composition, masterful lighting",
            "cartoon": "cartoon style, animated, colorful, fun, vibrant",
            "minimalist": "minimalist, clean, simple, modern design, elegant",
            "dramatic": "dramatic lighting, moody, atmospheric, intense",
            "funny": "humorous, comedic, lighthearted, playful, entertaining",
            "relatable": "everyday life, relatable, authentic, candid, natural"
        }
        
        style_desc = style_prompts.get(style, style_prompts["realistic"])
        
        # Clean the text
        clean_text = text.strip().replace('"', '').replace("'", "")
        
        # Add character consistency if available
        character_enhancement = ""
        if self.character_style:
            character_enhancement = f", {self.character_style['prompt_enhancement']}"
        
        enhanced_prompt = (
            f"{clean_text}. "
            f"Style: {style_desc}. "
            f"High resolution, well-lit, clear composition, centered subject, "
            f"9:16 aspect ratio, professional quality{character_enhancement}"
        )
        
        return enhanced_prompt
    
    def _get_default_negative_prompt(self) -> str:
        """Get default negative prompt to avoid common issues"""
        return (
            "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
            "watermark, signature, text, logo, multiple people, crowd, "
            "out of frame, poorly drawn, sketch, amateur, pixelated"
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
    
    def generate_thumbnail_for_video(self, title: str, style: str = "realistic") -> Optional[str]:
        """Generate a thumbnail image for the video"""
        thumbnail_prompt = f"YouTube thumbnail: {title}. Eye-catching, clickable thumbnail design, bold text, high contrast"
        return self.generate_image_from_text(thumbnail_prompt, style)
    
    def generate_video_from_image(
        self,
        image_path: str,
        motion_strength: float = 0.8,
        num_frames: int = 25,
        fps: int = 8,
        seed: Optional[int] = None,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.1
    ) -> Optional[str]:
        """
        Generate a motion video from a single image using Stable Video Diffusion
        
        Args:
            image_path: Path to the input image
            motion_strength: Strength of motion (0.0 to 1.0)
            num_frames: Number of frames to generate
            fps: Frames per second for the output video
            seed: Random seed for reproducibility
            motion_bucket_id: Motion bucket ID for different motion types
            noise_aug_strength: Noise augmentation strength
            
        Returns:
            Local path to the generated video, or None if failed
        """
        if self.video_pipeline is None:
            logger.error("Video pipeline not loaded. Cannot generate video.")
            return None
        
        try:
            logger.info(f"Generating video from image: {image_path}")
            
            # Set random seed
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Load and preprocess image
            image = load_image(image_path)
            
            # Ensure image is in the right format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image to standard size for video generation
            target_size = (1024, 576)  # 16:9 aspect ratio
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            logger.info(f"Processing image with size: {image.size}")
            
            # Generate video frames
            video_frames = self.video_pipeline(
                image,
                decode_chunk_size=8,
                motion_bucket_id=motion_bucket_id,
                fps=fps,
                noise_aug_strength=noise_aug_strength,
                num_frames=num_frames,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
            ).frames[0]
            
            logger.info(f"Generated {len(video_frames)} video frames")
            
            # Save video
            video_path = self._save_video(video_frames, image_path, seed, fps)
            
            if video_path:
                logger.info(f"Video generated and saved: {video_path}")
                return video_path
            else:
                logger.error("Failed to save video")
                return None
                
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            return None
    
    def generate_videos_for_script(
        self,
        script_lines: List[str],
        style: str = "realistic",
        motion_strength: float = 0.8,
        num_frames: int = 25,
        fps: int = 8,
        maintain_character_consistency: bool = True
    ) -> List[Optional[str]]:
        """
        Generate motion videos for each script line
        
        Args:
            script_lines: List of script text lines
            style: Image style for generation
            motion_strength: Strength of motion in videos
            num_frames: Number of frames per video
            fps: Frames per second
            maintain_character_consistency: Whether to maintain character consistency
            
        Returns:
            List of local paths for generated videos
        """
        video_paths = []
        
        for i, line in enumerate(script_lines):
            logger.info(f"Generating video {i+1}/{len(script_lines)} for: {line[:50]}...")
            
            # First generate an image for this line
            seed = None
            if maintain_character_consistency and self.character_seed is not None:
                seed = self.character_seed + i
            
            image_path = self.generate_image_from_text(
                text=line,
                style=style,
                seed=seed
            )
            
            if image_path:
                # Generate video from the image
                video_path = self.generate_video_from_image(
                    image_path=image_path,
                    motion_strength=motion_strength,
                    num_frames=num_frames,
                    fps=fps,
                    seed=seed
                )
                video_paths.append(video_path)
            else:
                logger.error(f"Failed to generate image for line {i+1}")
                video_paths.append(None)
            
            # Small delay to prevent memory issues
            import time
            time.sleep(1.0)
        
        return video_paths
    
    def generate_consistent_character_videos(
        self,
        script_lines: List[str],
        character_description: str,
        style: str = "realistic",
        motion_strength: float = 0.8,
        num_frames: int = 25,
        fps: int = 8,
        num_variations: int = 1
    ) -> List[List[Optional[str]]]:
        """
        Generate multiple variations of consistent character videos
        
        Args:
            script_lines: List of script text lines
            character_description: Description of the character
            style: Image style
            motion_strength: Strength of motion in videos
            num_frames: Number of frames per video
            fps: Frames per second
            num_variations: Number of variations to generate
            
        Returns:
            List of lists, where each inner list contains video paths for one variation
        """
        # Set up character consistency
        self.set_character_consistency(character_description)
        
        all_variations = []
        
        for variation in range(num_variations):
            logger.info(f"Generating video variation {variation + 1}/{num_variations}")
            
            # Generate base seed for this variation
            base_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
            variation_paths = []
            for i, line in enumerate(script_lines):
                # Use base seed + frame index for consistency within variation
                frame_seed = base_seed + i
                
                # Generate image first
                image_path = self.generate_image_from_text(
                    text=line,
                    style=style,
                    seed=frame_seed
                )
                
                if image_path:
                    # Generate video from image
                    video_path = self.generate_video_from_image(
                        image_path=image_path,
                        motion_strength=motion_strength,
                        num_frames=num_frames,
                        fps=fps,
                        seed=frame_seed
                    )
                    variation_paths.append(video_path)
                else:
                    variation_paths.append(None)
                
                import time
                time.sleep(1.0)
            
            all_variations.append(variation_paths)
        
        return all_variations
    
    def _save_video(self, video_frames: List[np.ndarray], original_image_path: str, 
                   seed: Optional[int] = None, fps: int = 8) -> Optional[str]:
        """Save the generated video frames to disk"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            
            # Extract description from original image path
            original_filename = os.path.basename(original_image_path)
            description = original_filename.split('_', 2)[-1].replace('.png', '') if '_' in original_filename else "video"
            
            seed_suffix = f"_seed{seed}" if seed else ""
            filename = f"{timestamp}_{unique_id}_{description}{seed_suffix}.mp4"
            
            # Save to video output directory
            output_path = os.path.join(self.video_output_dir, filename)
            
            # Convert frames to video
            export_to_video(video_frames, output_path, fps=fps)
            
            logger.info(f"Video saved to: {output_path}")
            return output_path
                
        except Exception as e:
            logger.error(f"Error saving video: {e}")
            return None
    
    def create_motion_video_from_image_sequence(
        self,
        image_paths: List[str],
        output_path: str,
        motion_strength: float = 0.8,
        num_frames_per_image: int = 15,
        fps: int = 8,
        transition_frames: int = 5
    ) -> Optional[str]:
        """
        Create a motion video from a sequence of images with smooth transitions
        
        Args:
            image_paths: List of image paths to convert to video
            output_path: Output video path
            motion_strength: Strength of motion in each segment
            num_frames_per_image: Number of frames to generate per image
            fps: Frames per second
            transition_frames: Number of transition frames between images
            
        Returns:
            Path to the generated video, or None if failed
        """
        if self.video_pipeline is None:
            logger.error("Video pipeline not loaded. Cannot generate video.")
            return None
        
        try:
            logger.info(f"Creating motion video from {len(image_paths)} images")
            
            all_video_frames = []
            
            for i, image_path in enumerate(image_paths):
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                
                # Generate video frames for this image
                video_frames = self.generate_video_from_image(
                    image_path=image_path,
                    motion_strength=motion_strength,
                    num_frames=num_frames_per_image,
                    fps=fps,
                    seed=i * 1000  # Use different seed for each image
                )
                
                if video_frames:
                    # Load the generated video frames
                    import cv2
                    cap = cv2.VideoCapture(video_frames)
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                    cap.release()
                    
                    all_video_frames.extend(frames)
                    
                    # Add transition frames if not the last image
                    if i < len(image_paths) - 1 and transition_frames > 0:
                        # Create simple fade transition
                        for t in range(transition_frames):
                            alpha = t / transition_frames
                            # Simple crossfade (you could implement more sophisticated transitions)
                            transition_frame = frames[-1]  # Use last frame as transition
                            all_video_frames.append(transition_frame)
            
            if all_video_frames:
                # Save combined video
                import cv2
                height, width = all_video_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                for frame in all_video_frames:
                    out.write(frame)
                
                out.release()
                
                logger.info(f"Motion video created: {output_path}")
                return output_path
            else:
                logger.error("No video frames generated")
                return None
                
        except Exception as e:
            logger.error(f"Error creating motion video: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.pipeline is not None:
            del self.pipeline
            logger.info("Image pipeline cleaned up")
        
        if self.video_pipeline is not None:
            del self.video_pipeline
            logger.info("Video pipeline cleaned up")
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None 