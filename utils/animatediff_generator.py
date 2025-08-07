#!/usr/bin/env python3
"""
Enhanced AnimateDiff Generator - Following Official Hugging Face Patterns

This implementation follows the official AnimateDiff documentation patterns:
- Proper MotionAdapter integration
- Memory optimization with chunking
- Better parameter handling
- Robust error handling

Based on: https://huggingface.co/docs/diffusers/en/api/pipelines/animatediff
"""

import os
import logging
import torch
import tempfile
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
from PIL import Image
import json
from dotenv import load_dotenv
import traceback

# Import huggingface_hub for downloading scheduler configs
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

# AnimateDiff imports
try:
    from diffusers import AnimateDiffPipeline, DDIMScheduler, DEISMultistepScheduler
    # Try different import paths for MotionAdapter
    try:
        from diffusers import MotionAdapter
    except ImportError:
        try:
            from diffusers.models.motion_adapter import MotionAdapter
        except ImportError:
            # If MotionAdapter is not available, we'll use the motion_adapter_path approach
            MotionAdapter = None
    ANIMATEDIFF_AVAILABLE = True
    print("✅ AnimateDiffPipeline import succeeded.")
except Exception as e:
    ANIMATEDIFF_AVAILABLE = False
    print("❌ AnimateDiff import failed.")
    traceback.print_exc()

load_dotenv()
logger = logging.getLogger(__name__)

class AnimateDiffGenerator:
    def __init__(self, 
                 sd_model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE",
                 motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5",  # Using public motion adapter
                 device: str = "auto",
                 memory_optimization: bool = True,
                 cache_dir: str = "models_cache"):
        """
        Initialize Enhanced AnimateDiff generator following official patterns
        
        Args:
            sd_model_id: Hugging Face model ID for Stable Diffusion
            motion_adapter_id: MotionAdapter checkpoint ID (from guoyww namespace)
            device: Device to run on ('auto', 'cuda', 'cpu')
            memory_optimization: Enable memory optimizations
            cache_dir: Directory to cache downloaded models
        """
        self.sd_model_id = sd_model_id
        self.motion_adapter_id = motion_adapter_id
        self.device = self._get_device(device)
        self.memory_optimization = memory_optimization
        self.cache_dir = cache_dir
        
        # Character consistency settings
        self.character_seed = None
        self.character_embeddings = {}
        self.character_style = {}
        
        # Frame limits for text-to-video generation (following official recommendations)
        self.min_frames = 16
        self.max_frames = 24  # AnimateDiff model/scheduler supports up to 24 frames
        self.default_frames = 24  # Use 24 as default for optimal compatibility
        
        # Memory optimization settings
        self.decode_chunk_size = 8  # Official recommendation for memory efficiency
        
        # Create output directories
        self.output_dir = "output/images"
        self.video_output_dir = "output/videos"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Using model cache directory: {os.path.abspath(self.cache_dir)}")
        
        # Initialize pipelines
        self.sd_pipeline = None
        self.animatediff_pipeline = None
        self._load_pipelines()
        
        logger.info(f"AnimateDiffGenerator initialized")
        logger.info(f"SD model: {sd_model_id}")
        logger.info(f"MotionAdapter: {motion_adapter_id}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Memory optimization: {memory_optimization}")
    
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
        """Load pipelines following official AnimateDiff patterns"""
        try:
            # Load Stable Diffusion pipeline for text-to-image
            logger.info(f"Loading Stable Diffusion pipeline on {self.device}...")
            
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                self.sd_model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                cache_dir=self.cache_dir,  # Cache the model
                # Remove variant parameter to avoid fp16 issues
            )
            
            # Configure scheduler for better quality
            try:
                # Load scheduler config from the model's scheduler subfolder
                import json
                
                scheduler_config_path = os.path.join(self.cache_dir, "models--" + self.sd_model_id.replace("/", "--"), "scheduler", "scheduler_config.json")
                
                # If config doesn't exist in cache, download it
                if not os.path.exists(scheduler_config_path):
                    if hf_hub_download is not None:
                        scheduler_config_path = hf_hub_download(
                            repo_id=self.sd_model_id,
                            filename="scheduler/scheduler_config.json",
                            cache_dir=self.cache_dir
                        )
                    else:
                        raise ImportError("huggingface_hub not available for downloading scheduler config")
                
                with open(scheduler_config_path, 'r') as f:
                    scheduler_config = json.load(f)
                
                # Add final_sigmas_type only for diffusers >= 0.33
                try:
                    import diffusers
                    diffusers_version = diffusers.__version__
                    if diffusers_version >= "0.33.0":
                        scheduler_config["final_sigmas_type"] = "sigma_min"
                        logger.info(f"Added final_sigmas_type for diffusers {diffusers_version}")
                    else:
                        logger.info(f"Skipping final_sigmas_type for diffusers {diffusers_version} (requires >= 0.33)")
                except (ImportError, AttributeError):
                    logger.warning("Could not determine diffusers version, skipping final_sigmas_type")
                
                scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
                self.sd_pipeline.scheduler = scheduler
                logger.info("DPMSolverMultistepScheduler configured successfully")
            except Exception as e:
                logger.warning(f"Could not configure scheduler: {e}")
                # Try with default settings if custom configuration fails
                try:
                    scheduler = DPMSolverMultistepScheduler.from_config({})
                    self.sd_pipeline.scheduler = scheduler
                    logger.info("DPMSolverMultistepScheduler configured with default settings")
                except Exception as e2:
                    logger.warning(f"Could not configure scheduler with default settings: {e2}")
            
            # Move to device
            self.sd_pipeline = self.sd_pipeline.to(self.device)
            
            # Enable memory optimizations
            if self.memory_optimization:
                if hasattr(self.sd_pipeline, "enable_attention_slicing"):
                    self.sd_pipeline.enable_attention_slicing()
                if hasattr(self.sd_pipeline, "enable_vae_slicing"):
                    self.sd_pipeline.enable_vae_slicing()
            
            logger.info("Stable Diffusion pipeline loaded successfully")
            
            # Load AnimateDiff pipeline following official patterns
            if ANIMATEDIFF_AVAILABLE:
                logger.info(f"Loading AnimateDiff pipeline with MotionAdapter: {self.motion_adapter_id}")
                
                try:
                    # Load the motion adapter directly (following your approach)
                    logger.info(f"Loading MotionAdapter: {self.motion_adapter_id}")
                    motion_adapter = MotionAdapter.from_pretrained(self.motion_adapter_id, cache_dir=self.cache_dir)
                    logger.info(f"Motion adapter loaded: {self.motion_adapter_id}")
                    
                    # Load AnimateDiff with MotionAdapter using the motion_adapter argument
                    self.animatediff_pipeline = AnimateDiffPipeline.from_pretrained(
                        self.sd_model_id,
                        motion_adapter=motion_adapter,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        cache_dir=self.cache_dir
                    )
                    
                    # Configure scheduler for AnimateDiff (following your approach)
                    logger.info("Configuring DDIMScheduler for AnimateDiff")
                    scheduler = DDIMScheduler.from_pretrained(
                        self.sd_model_id, 
                        subfolder="scheduler", 
                        clip_sample=False, 
                        timestep_spacing="linspace", 
                        steps_offset=1
                    )
                    self.animatediff_pipeline.scheduler = scheduler
                    logger.info("DDIMScheduler configured for AnimateDiff")
                    
                    # Move to device
                    self.animatediff_pipeline = self.animatediff_pipeline.to(self.device)
                    
                    # Enable memory optimizations (following your approach)
                    if self.memory_optimization:
                        if hasattr(self.animatediff_pipeline, "enable_vae_slicing"):
                            self.animatediff_pipeline.enable_vae_slicing()
                        if hasattr(self.animatediff_pipeline, "enable_model_cpu_offload"):
                            self.animatediff_pipeline.enable_model_cpu_offload()
                    
                    # Test the pipeline with minimal parameters
                    logger.info("Testing AnimateDiff pipeline...")
                    test_result = self.animatediff_pipeline(
                        prompt="test",
                        num_frames=2,
                        num_inference_steps=5
                    )
                    
                    logger.info(f"AnimateDiff pipeline loaded successfully with MotionAdapter: {self.motion_adapter_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to load AnimateDiff pipeline: {e}")
                    self.animatediff_pipeline = None
            
            if self.animatediff_pipeline is None:
                logger.warning("AnimateDiff pipeline failed to load - motion generation will be disabled")
                logger.info("You can still use Stable Diffusion for image generation")
                
        except Exception as e:
            logger.error(f"Error loading pipelines: {e}")
            raise
    
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
        Generate image from text using Stable Diffusion
        
        Args:
            text: Text description for image generation
            style: Image style
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            negative_prompt: What to avoid in the image
            seed: Random seed for reproducibility
            
        Returns:
            Local path to the generated image, or None if failed
        """
        if self.sd_pipeline is None:
            logger.error("Stable Diffusion pipeline not available")
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
            image_path = self._save_image(image, text, seed)
            
            if image_path:
                logger.info(f"Image generated and saved: {image_path}")
                return image_path
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
        num_frames: int = None,
        fps: int = 12,  # Increased default FPS for smoother video
        motion_strength: float = 0.8,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        negative_prompt: str = None,
        seed: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate animated frames from text using AnimateDiff (following official patterns)
        
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
            decode_chunk_size: Number of frames to decode at a time (memory optimization)
            
        Returns:
            Local path to the directory containing generated frames, or None if failed
        """
        if not ANIMATEDIFF_AVAILABLE or self.animatediff_pipeline is None:
            logger.error("AnimateDiff not available. Cannot generate animated video.")
            return None
        
        try:
            # Use character seed if available
            if self.character_seed is not None and seed is None:
                seed = self.character_seed
            
            # Validate frame count
            if num_frames is None:
                num_frames = self.default_frames
            num_frames = self._validate_frame_count(num_frames)
            
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
            logger.info(f"Using {num_frames} frames")
            
            # Generate animated video following your approach
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
            
            logger.info(f"Generated {len(video_frames)} video frames")
            
            # Save frames
            frames_dir = self._save_frames(video_frames, text, seed, fps)
            
            if frames_dir:
                logger.info(f"Animated frames generated and saved: {frames_dir}")
                return frames_dir
            else:
                logger.error("Failed to save animated frames")
                return None
                
        except Exception as e:
            logger.error(f"Error generating animated video: {e}")
            return None
    
    def _validate_frame_count(self, num_frames: int) -> int:
        """Validate frame count within acceptable range for AnimateDiff compatibility"""
        if num_frames < self.min_frames:
            logger.warning(f"Frame count {num_frames} too low, using minimum {self.min_frames}")
            return self.min_frames
        elif num_frames > self.max_frames:
            logger.warning(f"Frame count {num_frames} exceeds AnimateDiff model limit of {self.max_frames}, using maximum {self.max_frames}")
            return self.max_frames
        
        # Ensure frame count is compatible with motion adapter (multiples of 8)
        # AnimateDiff motion adapters work best with 16 or 24 frames
        if num_frames % 8 != 0:
            # Round to nearest multiple of 8
            adjusted_frames = round(num_frames / 8) * 8
            if adjusted_frames < self.min_frames:
                adjusted_frames = self.min_frames
            elif adjusted_frames > self.max_frames:
                adjusted_frames = self.max_frames
            logger.info(f"Adjusted frame count from {num_frames} to {adjusted_frames} for motion adapter compatibility")
            return adjusted_frames
        
        return num_frames
    
    def _create_enhanced_prompt(self, text: str, style: str) -> str:
        """Create enhanced prompt with style and quality modifiers"""
        base_prompt = text.strip()
        
        # Add style-specific enhancements
        style_enhancements = {
            "realistic": "high quality, detailed, realistic, professional photography",
            "anime": "anime style, high quality, detailed, vibrant colors",
            "cinematic": "cinematic lighting, dramatic, high quality, professional cinematography",
            "artistic": "artistic, creative, high quality, detailed artwork",
            "sci-fi": "sci-fi, futuristic, high quality, detailed, advanced technology",
            "cartoon": "cartoon style, cute, child-friendly, soft colors, rounded shapes, safe for children",
            "cute": "cute, adorable, child-friendly, soft colors, rounded shapes, safe for children",
            "bright": "bright, vibrant, colorful, cheerful, energetic, child-friendly",
            "simple": "simple, clean, minimalist, easy to understand, child-friendly",
            "whimsical": "whimsical, magical, fantastical, dreamy, child-friendly",
            "educational": "clear, educational, informative, engaging, child-friendly"
        }
        
        enhancement = style_enhancements.get(style.lower(), "high quality, detailed")
        
        # Combine prompt with enhancement
        enhanced_prompt = f"{base_prompt}, {enhancement}"
        
        return enhanced_prompt
    
    def _get_default_negative_prompt(self) -> str:
        """Get default negative prompt for better quality"""
        return "low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, signature"
    
    def _save_image(self, image: Image.Image, text: str, seed: Optional[int] = None) -> Optional[str]:
        """Save generated image with metadata"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            safe_text = "".join(c for c in text[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_text = safe_text.replace(' ', '_')
            
            filename = f"{timestamp}_{unique_id}_{safe_text}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save image
            image.save(filepath, "PNG")
            
            # Save metadata
            metadata = {
                "text": text,
                "seed": seed,
                "timestamp": timestamp,
                "model": self.sd_model_id
            }
            
            metadata_path = filepath.replace(".png", "_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None
    
    def _save_frames(self, video_frames: List[Image.Image], original_text: str, 
                   seed: Optional[int] = None, fps: int = 8) -> Optional[str]:
        """Save generated video frames as individual PNG files and return frames directory path"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            safe_text = "".join(c for c in original_text[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_text = safe_text.replace(' ', '_')
            
            # Create temporary directory for frames
            frames_dir = os.path.join(self.video_output_dir, f"frames_{timestamp}_{unique_id}")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Save individual frames as PNG files
            frame_paths = []
            for i, frame in enumerate(video_frames):
                frame_filename = f"frame_{i+1:04d}.png"
                frame_path = os.path.join(frames_dir, frame_filename)
                frame.save(frame_path, "PNG")
                frame_paths.append(frame_path)
            
            logger.info(f"Saved {len(frame_paths)} frames to {frames_dir}")
            
            # Save metadata
            metadata = {
                "text": original_text,
                "seed": seed,
                "timestamp": timestamp,
                "fps": fps,
                "frames": len(video_frames),
                "motion_adapter": self.motion_adapter_id,
                "model": self.sd_model_id,
                "frames_directory": frames_dir
            }
            
            metadata_path = os.path.join(frames_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return frames_dir
            
        except Exception as e:
            logger.error(f"Error saving frames: {e}")
            return None
    
    def _compile_frames_to_video(self, frame_paths: List[str], output_path: str, fps: int = 8) -> bool:
        """Compile individual frames into a video using FFmpeg"""
        try:
            if not frame_paths:
                logger.error("No frame paths provided")
                return False
            
            # Get the directory containing the frames
            frames_dir = os.path.dirname(frame_paths[0])
            
            # Create FFmpeg command to compile frames
            # Using -framerate 8 (not -r) and libx264 with yuv420p pixel format
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-framerate', str(fps),  # Input frame rate
                '-i', os.path.join(frames_dir, 'frame_%04d.png'),  # Input pattern
                '-c:v', 'libx264',  # Video codec
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-preset', 'medium',  # Encoding preset (balance between speed and quality)
                '-crf', '23',  # Constant Rate Factor (quality setting, lower = better quality)
                output_path
            ]
            
            logger.info(f"Compiling frames to video: {' '.join(cmd)}")
            
            # Run FFmpeg command
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully compiled video: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg failed with return code {result.returncode}")
                logger.error(f"FFmpeg stderr: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error compiling frames to video: {e}")
            return False
    
    def compile_multiple_frame_directories_to_video(self, frame_directories: List[str], output_path: str, fps: int = 8) -> bool:
        """Compile multiple frame directories into a single video by concatenating them"""
        try:
            if not frame_directories:
                logger.error("No frame directories provided")
                return False
            
            # Create temporary directory for combined frames
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                combined_frame_paths = []
                frame_counter = 1
                
                # Copy all frames from all directories to a single directory with sequential numbering
                for frame_dir in frame_directories:
                    if not os.path.exists(frame_dir):
                        logger.warning(f"Frame directory does not exist: {frame_dir}")
                        continue
                    
                    # Get all frame files in the directory
                    frame_files = sorted([f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.png')])
                    
                    for frame_file in frame_files:
                        # Copy frame to combined directory with new sequential name
                        new_frame_name = f"frame_{frame_counter:04d}.png"
                        new_frame_path = os.path.join(temp_dir, new_frame_name)
                        
                        import shutil
                        shutil.copy2(os.path.join(frame_dir, frame_file), new_frame_path)
                        combined_frame_paths.append(new_frame_path)
                        frame_counter += 1
                
                if not combined_frame_paths:
                    logger.error("No frames found in any of the provided directories")
                    return False
                
                logger.info(f"Combined {len(combined_frame_paths)} frames from {len(frame_directories)} directories")
                
                # Compile combined frames to video
                return self._compile_frames_to_video(combined_frame_paths, output_path, fps)
                
        except Exception as e:
            logger.error(f"Error compiling multiple frame directories to video: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.sd_pipeline:
                del self.sd_pipeline
            if self.animatediff_pipeline:
                del self.animatediff_pipeline
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("AnimateDiffGenerator cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

# Example usage function
def test_animatediff():
    """Test the enhanced AnimateDiff generator"""
    logger.info("Testing Enhanced AnimateDiff Generator")
    
    try:
        # Initialize generator
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-motion-adapter-v1-5",
            memory_optimization=True,
            cache_dir="models_cache"  # Cache models locally
        )
        
        # Test image generation
        logger.info("Testing image generation...")
        image_path = generator.generate_image_from_text(
            text="A beautiful sunset over mountains",
            style="realistic",
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=42
        )
        
        if image_path:
            logger.info(f"✅ Image generated: {image_path}")
        
        # Test video generation
        logger.info("Testing video generation...")
        video_path = generator.generate_animated_video_from_text(
            text="A cat sitting in a garden",
            style="realistic",
            num_frames=16,
            fps=8,
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=42
        )
        
        if video_path:
            logger.info(f"✅ Video generated: {video_path}")
        
        # Cleanup
        generator.cleanup()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    test_animatediff() 