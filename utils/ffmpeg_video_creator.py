import os
import ffmpeg
import tempfile
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
import json
import logging
from .stable_diffusion_generator import StableDiffusionGenerator
from .s3_uploader import S3Uploader

logger = logging.getLogger(__name__)

class FFmpegVideoCreator:
    def __init__(self, use_stable_video_diffusion: bool = True):
        """
        Initialize FFmpeg video creator with optional stable video diffusion
        
        Args:
            use_stable_video_diffusion: Whether to use stable video diffusion for motion
        """
        self.width = 1080
        self.height = 1920  # 9:16 aspect ratio for YouTube Shorts
        self.fps = 30
        self.use_stable_video_diffusion = use_stable_video_diffusion
        
        # Initialize stable diffusion generator if needed
        self.sd_generator = None
        if use_stable_video_diffusion:
            try:
                self.sd_generator = StableDiffusionGenerator()
                logger.info("Stable diffusion generator initialized for video creation")
            except Exception as e:
                logger.warning(f"Failed to initialize stable diffusion generator: {e}")
                self.use_stable_video_diffusion = False
        
        # Initialize S3 uploader
        self.s3_uploader = None
        try:
            self.s3_uploader = S3Uploader()
            logger.info("S3 uploader initialized for video uploads")
        except Exception as e:
            logger.warning(f"Failed to initialize S3 uploader: {e}")
            self.s3_uploader = None
    
    def create_video_with_motion(
        self, 
        audio_path: str, 
        narration_lines: List[dict], 
        output_path: str,
        motion_strength: float = 0.8,
        num_frames_per_segment: int = 25,
        video_fps: int = 8
    ) -> str:
        """
        Create a vertical video with audio, motion videos, and subtitles
        
        Args:
            audio_path: Path to audio file
            narration_lines: List of narration line dictionaries
            output_path: Output video path
            motion_strength: Strength of motion in stable video diffusion
            num_frames_per_segment: Number of frames per video segment
            video_fps: FPS for generated motion videos
            
        Returns:
            Path to the created video
        """
        logger.info(f"Creating video with motion from {len(narration_lines)} narration lines")
        
        # Create temporary directory for assets
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate motion videos or static images for each narration line
            video_paths = []
            subtitle_paths = []
            
            for i, line in enumerate(narration_lines):
                logger.info(f"Processing segment {i+1}/{len(narration_lines)}")
                
                # Create motion video or static image
                if self.use_stable_video_diffusion and self.sd_generator:
                    video_path = self._create_motion_video_segment(
                        temp_dir, line, i, motion_strength, num_frames_per_segment, video_fps
                    )
                else:
                    video_path = self._create_static_image_segment(temp_dir, line, i)
                
                video_paths.append(video_path)
                
                # Create subtitle image
                subtitle_path = os.path.join(temp_dir, f"subtitle_{i}.png")
                self._create_subtitle_image(subtitle_path, line["text"])
                subtitle_paths.append(subtitle_path)
            
            logger.info(f"Generated {len(video_paths)} video segments and {len(subtitle_paths)} subtitles")
            
            # Create final video using FFmpeg
            self._combine_assets_with_motion(
                audio_path, video_paths, subtitle_paths, narration_lines, output_path
            )
            
        return output_path
    
    def _create_motion_video_segment(
        self, 
        temp_dir: str, 
        line: dict, 
        index: int,
        motion_strength: float,
        num_frames: int,
        fps: int
    ) -> str:
        """Create a motion video segment using stable video diffusion"""
        try:
            # Get target duration from the line
            target_duration = line.get("duration", 3.0)
            logger.info(f"Creating motion video segment {index} with target duration: {target_duration}s")
            
            # For now, use static image to ensure exact duration control
            # This prevents the 36-minute video issue
            logger.info(f"Using static image for segment {index} to ensure exact duration control")
            return self._create_static_image_segment(temp_dir, line, index)
            
            # TODO: Re-enable motion video generation once duration issues are resolved
            # Generate image first
            # image_path = self.sd_generator.generate_image_from_text(
            #     text=line.get("visual_suggestion", line["text"]),
            #     style="realistic",
            #     seed=index * 1000
            # )
            # 
            # if image_path:
            #     # Generate motion video from the image with target duration
            #     video_path = self.sd_generator.generate_video_from_image(
            #         image_path=image_path,
            #         motion_strength=motion_strength,
            #         num_frames=num_frames,
            #         fps=fps,
            #         seed=index * 1000,
            #         target_duration=target_duration  # Pass target duration directly
            #     )
            #     
            #     if video_path:
            #         logger.info(f"Motion video segment {index} created: {video_path}")
            #         return video_path
            
            # Fallback to static image if video generation fails
            # logger.warning(f"Motion video generation failed for segment {index}, using static image")
            # return self._create_static_image_segment(temp_dir, line, index)
            
        except Exception as e:
            logger.error(f"Error creating motion video segment: {e}")
            return self._create_static_image_segment(temp_dir, line, index)
    
    def _create_static_image_segment(self, temp_dir: str, line: dict, index: int) -> str:
        """Create a static image segment as fallback"""
        image_path = os.path.join(temp_dir, f"static_image_{index}.png")
        self._create_placeholder_image(image_path, line.get("visual_suggestion", "placeholder"))
        return image_path
    
    def _create_placeholder_image(self, output_path: str, description: str):
        """Create a placeholder image with description"""
        try:
            img = Image.new('RGB', (self.width, self.height), color='#2C3E50')
            draw = ImageDraw.Draw(img)
            
            # Try multiple font options
            font = None
            font_size = 60
            font_options = [
                "arial.ttf",
                "Arial.ttf", 
                "C:/Windows/Fonts/arial.ttf",
                "/System/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ]
            
            for font_path in font_options:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    logger.info(f"Loaded font: {font_path}")
                    break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
                logger.warning("Using default font")
            
            # Add description text
            text = f"Image: {description}"
            
            # Wrap text if too long
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] - bbox[0] <= self.width - 100:  # Leave 50px margin on each side
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Draw each line
            line_height = font_size + 10
            total_height = len(lines) * line_height
            start_y = (self.height - total_height) // 2
            
            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x = (self.width - text_width) // 2
                y = start_y + i * line_height
                draw.text((x, y), line, fill='white', font=font)
            
            img.save(output_path)
            logger.info(f"Created placeholder image: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating placeholder image: {e}")
            # Create a simple fallback image
            img = Image.new('RGB', (self.width, self.height), color='#34495E')
            draw = ImageDraw.Draw(img)
            draw.text((self.width//2, self.height//2), "Why Would You", fill='white')
            img.save(output_path)
    
    def _create_subtitle_image(self, output_path: str, text: str):
        """Create a subtitle image with text"""
        try:
            # Create a full-size transparent image
            img = Image.new('RGBA', (self.width, self.height), color=(0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Try multiple font options
            font = None
            font_size = 48
            font_options = [
                "arial.ttf",
                "Arial.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "/System/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ]
            
            for font_path in font_options:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
            
            # Wrap text if too long
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] - bbox[0] <= self.width - 100:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Calculate total height needed
            line_height = font_size + 10
            total_height = len(lines) * line_height
            padding = 20
            subtitle_height = total_height + 2 * padding
            
            # Position subtitles at bottom of screen
            subtitle_y = self.height - subtitle_height - 100  # 100px from bottom
            
            # Draw background rectangle for all lines
            bbox = draw.textbbox((0, 0), " ".join(lines), font=font)
            text_width = bbox[2] - bbox[0]
            
            x = (self.width - text_width) // 2
            y = subtitle_y
            
            # Draw background
            draw.rectangle([
                x - padding, y - padding,
                x + text_width + padding, y + subtitle_height - padding
            ], fill=(0, 0, 0, 180))
            
            # Draw each line
            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                line_x = (self.width - line_width) // 2
                line_y = subtitle_y + i * line_height
                draw.text((line_x, line_y), line, fill='white', font=font)
            
            img.save(output_path)
            logger.info(f"Created subtitle image: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating subtitle image: {e}")
            # Create a simple fallback subtitle
            img = Image.new('RGBA', (self.width, self.height), color=(0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((self.width//2, self.height-150), text[:50], fill='white')
            img.save(output_path)
    
    def _combine_assets_with_motion(
        self, 
        audio_path: str, 
        video_paths: List[str], 
        subtitle_paths: List[str], 
        narration_lines: List[dict], 
        output_path: str
    ):
        """Combine audio, motion videos, and subtitles into final video"""
        
        try:
            # Get audio duration
            audio_duration = 30.0  # Default fallback duration
            try:
                probe = ffmpeg.probe(audio_path)
                if probe and 'streams' in probe and len(probe['streams']) > 0:
                    if 'duration' in probe['streams'][0]:
                        audio_duration = float(probe['streams'][0]['duration'])
                    else:
                        # Try to get duration from format info
                        if 'format' in probe and 'duration' in probe['format']:
                            audio_duration = float(probe['format']['duration'])
                logger.info(f"Audio duration: {audio_duration} seconds")
            except Exception as e:
                logger.warning(f"Could not probe audio duration: {e}, using default {audio_duration}s")
            
            # Create video stream from motion videos or images
            video_inputs = []
            current_time = 0
            
            for i, (video_path, subtitle_path, line) in enumerate(zip(video_paths, subtitle_paths, narration_lines)):
                duration = line.get("duration", 3.0)
                logger.info(f"Processing segment {i+1}: duration={duration}s")
                
                # Check if it's a video file or image
                if video_path.endswith('.mp4'):
                    logger.info(f"It's a motion video===============")
                    video_stream = (
                        ffmpeg
                        .input(video_path)
                        .filter('scale', self.width, self.height)
                        .filter('fps', fps=self.fps)
                        .filter('loop', loop=1, size=1)  # Loop the video for the duration
                        .filter('trim', duration=duration)
                    )
                else:
                    # It's a static image
                    video_stream = (
                        ffmpeg
                        .input(video_path, loop=1, t=duration)
                        .filter('scale', self.width, self.height)
                    )
                
                # Create subtitle stream
                subtitle_stream = (
                    ffmpeg
                    .input(subtitle_path, loop=1, t=duration)
                    .filter('scale', self.width, self.height)
                )
                
                # Overlay subtitle on video
                combined = ffmpeg.overlay(video_stream, subtitle_stream, x=0, y=0)
                video_inputs.append(combined)
                
                current_time += duration
            
            # Concatenate all video segments
            if len(video_inputs) > 1:
                video = ffmpeg.concat(*video_inputs, v=1, a=0)
            else:
                video = video_inputs[0]
            
            # Add audio
            audio = ffmpeg.input(audio_path)
            
            # Output final video
            logger.info(f"Rendering final video to: {output_path}")
            (
                ffmpeg
                .output(video, audio, output_path,
                       vcodec='libx264', acodec='aac',
                       pix_fmt='yuv420p', r=self.fps,
                       video_bitrate='2M', audio_bitrate='128k')
                .overwrite_output()
                .run(quiet=True)
            )
            
            logger.info(f"Video created successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Error combining assets: {e}")
            # Fallback to simple video creation
            self.create_simple_video(audio_path, output_path, audio_duration)
    
    def create_video_with_images(
        self, 
        audio_path: str, 
        narration_lines: List[dict], 
        image_paths: List[str], 
        output_path: str,
        animation_type: str = "zoom_in"
    ) -> str:
        """
        Create a video with audio, images, and subtitles
        
        Args:
            audio_path: Path to audio file
            narration_lines: List of narration line dictionaries
            image_paths: List of image paths
            output_path: Output video path
            animation_type: Type of animation (zoom_in, pan, fade, etc.)
            
        Returns:
            Path to the created video
        """
        logger.info(f"Creating video with {len(image_paths)} images and {len(narration_lines)} narration lines")
        
        # Create temporary directory for assets
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create subtitle images
            subtitle_paths = []
            for i, line in enumerate(narration_lines):
                subtitle_path = os.path.join(temp_dir, f"subtitle_{i}.png")
                self._create_subtitle_image(subtitle_path, line["text"])
                subtitle_paths.append(subtitle_path)
            
            # Create final video using FFmpeg
            self._combine_images_with_audio(
                audio_path, image_paths, subtitle_paths, narration_lines, output_path, animation_type
            )
            
        return output_path
    
    def _combine_images_with_audio(
        self, 
        audio_path: str, 
        image_paths: List[str], 
        subtitle_paths: List[str], 
        narration_lines: List[dict], 
        output_path: str,
        animation_type: str
    ):
        """Combine images, audio, and subtitles into final video"""
        
        try:
            # Get audio duration
            audio_duration = 30.0  # Default fallback duration
            try:
                probe = ffmpeg.probe(audio_path)
                if probe and 'streams' in probe and len(probe['streams']) > 0:
                    if 'duration' in probe['streams'][0]:
                        audio_duration = float(probe['streams'][0]['duration'])
                    else:
                        # Try to get duration from format info
                        if 'format' in probe and 'duration' in probe['format']:
                            audio_duration = float(probe['format']['duration'])
                logger.info(f"Audio duration: {audio_duration} seconds")
            except Exception as e:
                logger.warning(f"Could not probe audio duration: {e}, using default {audio_duration}s")
            
            # Create video stream from images
            video_inputs = []
            current_time = 0
            
            # Ensure we have the same number of images and narration lines
            min_count = min(len(image_paths), len(subtitle_paths), len(narration_lines))
            logger.info(f"Processing {min_count} segments (images: {len(image_paths)}, subtitles: {len(subtitle_paths)}, narration: {len(narration_lines)})")
            
            for i in range(min_count):
                image_path = image_paths[i]
                subtitle_path = subtitle_paths[i]
                line = narration_lines[i]
                duration = line.get("duration", 3.0)
                logger.info(f"Processing segment {i+1}: duration={duration}s, image={image_path}")
                
                # Apply animation filter based on type
                try:
                    if animation_type == "zoom_in":
                        # Calculate frames for zoompan duration
                        zoom_frames = int(duration * self.fps)
                        video_stream = (
                            ffmpeg
                            .input(image_path, loop=1, t=duration)
                            .filter('scale', self.width, self.height)
                            .filter('zoompan', z='min(zoom+0.0015,1.5)', d=zoom_frames, x='iw/2-(iw/zoom/2)', y='ih/2-(ih/zoom/2)')
                        )
                    elif animation_type == "pan":
                        # Calculate frames for pan duration
                        pan_frames = int(duration * self.fps)
                        video_stream = (
                            ffmpeg
                            .input(image_path, loop=1, t=duration)
                            .filter('scale', self.width * 1.2, self.height * 1.2)
                            .filter('crop', self.width, self.height, f't*{pan_frames//10}', f't*{pan_frames//15}')
                        )
                    elif animation_type == "fade":
                        video_stream = (
                            ffmpeg
                            .input(image_path, loop=1, t=duration)
                            .filter('scale', self.width, self.height)
                            .filter('fade', t='in', st=0, d=0.5)
                            .filter('fade', t='out', st=duration-0.5, d=0.5)
                        )
                    else:
                        # Default: static image
                        video_stream = (
                            ffmpeg
                            .input(image_path, loop=1, t=duration)
                            .filter('scale', self.width, self.height)
                        )
                except Exception as e:
                    logger.warning(f"Animation filter {animation_type} failed: {e}, using static image")
                    # Fallback to static image
                    video_stream = (
                        ffmpeg
                        .input(image_path, loop=1, t=duration)
                        .filter('scale', self.width, self.height)
                    )
                
                # Create subtitle stream
                subtitle_stream = (
                    ffmpeg
                    .input(subtitle_path, loop=1, t=duration)
                    .filter('scale', self.width, self.height)
                )
                
                # Overlay subtitle on video
                combined = ffmpeg.overlay(video_stream, subtitle_stream, x=0, y=0)
                video_inputs.append(combined)
                
                current_time += duration
            
            # Concatenate all video segments
            if len(video_inputs) > 1:
                video = ffmpeg.concat(*video_inputs, v=1, a=0)
            else:
                video = video_inputs[0]
            
            # Add audio
            logger.info(f"Audio path: {audio_path}")
            audio = ffmpeg.input(audio_path)
            
            # Output final video
            logger.info(f"Rendering final video to: {output_path}")
            (
                ffmpeg.output(
                    video, audio, output_path,
                    vcodec='h264_nvenc',           # GPU encoder
                    acodec='aac',
                    pix_fmt='yuv420p',
                    r=self.fps,
                    video_bitrate='4M',            # You can tweak this
                    audio_bitrate='128k',
                    **{'preset': 'fast'}           # Fastest reliable preset for NVENC
                ).overwrite_output().run(quiet=True)
            )
            
            logger.info(f"Video created successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Error combining images with audio: {e}")
            # Fallback to simple video creation
            self.create_simple_video(audio_path, output_path, audio_duration)
    
    def create_simple_video(self, audio_path: str, output_path: str, duration: float = 30.0) -> str:
        """Create a simple video with just audio and a static background"""
        
        try:
            # Create a simple background image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
                img = Image.new('RGB', (self.width, self.height), color='#34495E')
                draw = ImageDraw.Draw(img)
                
                # Add channel name
                font = None
                font_options = [
                    "arial.ttf",
                    "Arial.ttf",
                    "C:/Windows/Fonts/arial.ttf",
                    "/System/Library/Fonts/Arial.ttf"
                ]
                
                for font_path in font_options:
                    try:
                        font = ImageFont.truetype(font_path, 80)
                        break
                    except:
                        continue
                
                if font is None:
                    font = ImageFont.load_default()
                
                text = "Why Would You"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                x = (self.width - text_width) // 2
                y = (self.height - text_height) // 2
                
                draw.text((x, y), text, fill='white', font=font)
                img.save(tmp_img.name)
                
                # Create video
                logger.info(f"Creating simple video with background image")
                process = (
                    ffmpeg
                    .output(video, audio, output_path,
                            vcodec='h264_nvenc',
                            acodec='aac',
                            pix_fmt='yuv420p',
                            r=30,
                            video_bitrate='4M',
                            audio_bitrate='128k',
                            shortest=None,
                            **{'preset': 'fast'})
                    .overwrite_output()
                    .run_async(pipe_stderr=True)
                )

                # Print progress lines in real-time
                for line in process.stderr:
                    logger.info(line.decode('utf-8').strip())
                
                # Clean up
                os.unlink(tmp_img.name)
            
            logger.info(f"Simple video created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating simple video: {e}")
            raise
    
    def upload_to_s3(self, video_path: str, s3_key: str = None, folder: str = "youtube-shorts") -> Optional[str]:
        """
        Upload the final video to S3 bucket
        
        Args:
            video_path: Path to the local video file
            s3_key: Custom S3 key (optional, will generate if not provided)
            folder: S3 folder path (default: youtube-shorts)
            
        Returns:
            S3 URL of uploaded video or None if failed
        """
        if not self.s3_uploader:
            logger.error("S3 uploader not initialized")
            return None
        
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return None
        
        try:
            logger.info(f"Uploading video to S3: {video_path}")
            s3_url = self.s3_uploader.upload_video(video_path, s3_key, folder)
            
            if s3_url:
                logger.info(f"Video successfully uploaded to S3: {s3_url}")
                return s3_url
            else:
                logger.error("Failed to upload video to S3")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading video to S3: {e}")
            return None
    
    def create_and_upload_video(
        self,
        audio_path: str,
        narration_lines: List[dict],
        image_paths: List[str] = None,
        output_path: str = None,
        upload_to_s3: bool = True,
        s3_folder: str = "youtube-shorts"
    ) -> dict:
        """
        Create video and optionally upload to S3
        
        Args:
            audio_path: Path to audio file
            narration_lines: List of narration line dictionaries
            image_paths: List of image paths (optional, for image-based videos)
            output_path: Output video path (optional, will generate if not provided)
            upload_to_s3: Whether to upload to S3 after creation
            s3_folder: S3 folder path
            
        Returns:
            Dictionary with local_path and s3_url (if uploaded)
        """
        result = {"local_path": None, "s3_url": None}
        
        try:
            # Generate output path if not provided
            if not output_path:
                import time
                timestamp = int(time.time())
                output_path = f"output/shorts_with_images_{timestamp}.mp4"
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create video
            if image_paths:
                logger.info("Creating video with images")
                result["local_path"] = self.create_video_with_images(
                    audio_path, narration_lines, image_paths, output_path
                )
            else:
                logger.info("Creating video with motion")
                result["local_path"] = self.create_video_with_motion(
                    audio_path, narration_lines, output_path
                )
            
            # Upload to S3 if requested
            if upload_to_s3 and result["local_path"]:
                logger.info("Uploading video to S3")
                result["s3_url"] = self.upload_to_s3(result["local_path"], folder=s3_folder)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in create_and_upload_video: {e}")
            return result
    
    def create_video_with_motion_images(
        self, 
        audio_path: str, 
        narration_lines: List[dict], 
        image_paths: List[str], 
        output_path: str,
        motion_strength: float = 1.0,  # Increased from 0.8 to 1.0 for more visible motion
        num_frames_per_segment: int = 12,  # Reduced for faster generation
        video_fps: int = 8,
        fast_mode: bool = True,  # Enable fast mode by default
        timeout_seconds: int = 60,  # Timeout for each motion video generation
        motion_type: str = "dynamic"  # Type of motion to generate
    ) -> str:
        """
        Create a video with audio, motion videos generated from images, and subtitles
        
        Args:
            audio_path: Path to audio file
            narration_lines: List of narration line dictionaries
            image_paths: List of image paths to convert to motion videos
            output_path: Output video path
            motion_strength: Strength of motion in stable video diffusion (increased for more visible motion)
            num_frames_per_segment: Number of frames per video segment (reduced for speed)
            video_fps: FPS for generated motion videos
            fast_mode: Use fast mode for quicker generation
            timeout_seconds: Timeout for each motion video generation
            motion_type: Type of motion to generate ("dynamic", "camera_movement", "object_motion", etc.)
            
        Returns:
            Path to the created video
        """
        logger.info(f"Creating video with motion from {len(image_paths)} images and {len(narration_lines)} narration lines")
        logger.info(f"Enhanced motion settings: strength={motion_strength}, type={motion_type}, fast_mode={fast_mode}, timeout={timeout_seconds}s per segment")
        
        if not self.sd_generator:
            logger.warning("Stable diffusion generator not available, falling back to static images")
            return self.create_video_with_images(audio_path, narration_lines, image_paths, output_path)
        
        # Create temporary directory for assets
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate motion videos from images
            video_paths = []
            subtitle_paths = []
            motion_videos_created = 0
            static_images_used = 0
            
            for i, (image_path, line) in enumerate(zip(image_paths, narration_lines)):
                logger.info(f"Processing segment {i+1}/{len(image_paths)}")
                
                # Get target duration from the line
                target_duration = line.get("duration", 3.0)
                logger.info(f"Creating motion video for image {i+1} with target duration: {target_duration}s")
                
                # Try to generate motion video with timeout
                video_path = None
                try:
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Motion video generation timed out after {timeout_seconds} seconds")
                    
                    # Set timeout
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_seconds)
                    
                    # Generate motion video from the image with enhanced motion settings
                    video_path = self.sd_generator.generate_video_from_image(
                        image_path=image_path,
                        motion_strength=motion_strength,  # Enhanced motion strength
                        num_frames=num_frames_per_segment,
                        fps=video_fps,
                        seed=i * 1000,
                        target_duration=target_duration,
                        fast_mode=fast_mode,
                        motion_type=motion_type,  # Use dynamic motion type
                        noise_aug_strength=0.3  # Increased for more visible motion
                    )
                    
                    # Cancel timeout
                    signal.alarm(0)
                    
                except TimeoutError:
                    logger.warning(f"Motion video generation timed out for segment {i+1}, using static image")
                    video_path = None
                except Exception as e:
                    logger.error(f"Error generating motion video for segment {i+1}: {e}")
                    video_path = None
                
                if video_path:
                    video_paths.append(video_path)
                    motion_videos_created += 1
                    logger.info(f"Motion video {i+1} created: {video_path}")
                else:
                    # Fallback to static image
                    logger.warning(f"Motion video generation failed for segment {i}, using static image")
                    video_paths.append(image_path)
                    static_images_used += 1
                
                # Create subtitle image
                subtitle_path = os.path.join(temp_dir, f"subtitle_{i}.png")
                self._create_subtitle_image(subtitle_path, line["text"])
                subtitle_paths.append(subtitle_path)
            
            logger.info(f"Generated {len(video_paths)} video segments: {motion_videos_created} motion videos, {static_images_used} static images")
            
            # Create final video using FFmpeg
            self._combine_assets_with_motion(
                audio_path, video_paths, subtitle_paths, narration_lines, output_path
            )
            
        return output_path
    
    def create_video_with_ffmpeg_motion(
        self, 
        audio_path: str, 
        narration_lines: List[dict], 
        image_paths: List[str], 
        output_path: str,
        motion_type: str = "zoom_pan"  # zoom_pan, rotate, scale, etc.
    ) -> str:
        """
        Create a video with audio, FFmpeg motion effects on images, and subtitles
        This is much faster than Stable Video Diffusion but less sophisticated
        
        Args:
            audio_path: Path to audio file
            narration_lines: List of narration line dictionaries
            image_paths: List of image paths to apply motion effects to
            output_path: Output video path
            motion_type: Type of motion effect (zoom_pan, rotate, scale, etc.)
            
        Returns:
            Path to the created video
        """
        logger.info(f"Creating video with FFmpeg motion effects from {len(image_paths)} images")
        
        # Create temporary directory for assets
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create subtitle images
            subtitle_paths = []
            for i, line in enumerate(narration_lines):
                subtitle_path = os.path.join(temp_dir, f"subtitle_{i}.png")
                self._create_subtitle_image(subtitle_path, line["text"])
                subtitle_paths.append(subtitle_path)
            
            # Create final video using FFmpeg with motion effects
            self._combine_images_with_ffmpeg_motion(
                audio_path, image_paths, subtitle_paths, narration_lines, output_path, motion_type
            )
            
        return output_path
    
    def _combine_images_with_ffmpeg_motion(
        self, 
        audio_path: str, 
        image_paths: List[str], 
        subtitle_paths: List[str], 
        narration_lines: List[dict], 
        output_path: str,
        motion_type: str
    ):
        """Combine images with FFmpeg motion effects, audio, and subtitles into final video"""
        
        try:
            # Get audio duration
            audio_duration = 30.0  # Default fallback duration
            try:
                probe = ffmpeg.probe(audio_path)
                if probe and 'streams' in probe and len(probe['streams']) > 0:
                    if 'duration' in probe['streams'][0]:
                        audio_duration = float(probe['streams'][0]['duration'])
                    else:
                        # Try to get duration from format info
                        if 'format' in probe and 'duration' in probe['format']:
                            audio_duration = float(probe['format']['duration'])
                logger.info(f"Audio duration: {audio_duration} seconds")
            except Exception as e:
                logger.warning(f"Could not probe audio duration: {e}, using default {audio_duration}s")
            
            # Create video stream from images with motion effects
            video_inputs = []
            current_time = 0
            
            # Ensure we have the same number of images and narration lines
            min_count = min(len(image_paths), len(subtitle_paths), len(narration_lines))
            logger.info(f"Processing {min_count} segments with FFmpeg motion effects")
            
            for i in range(min_count):
                image_path = image_paths[i]
                subtitle_path = subtitle_paths[i]
                line = narration_lines[i]
                duration = line.get("duration", 3.0)
                logger.info(f"Processing segment {i+1}: duration={duration}s, motion={motion_type}")
                
                # Apply FFmpeg motion effects based on type
                try:
                    if motion_type == "zoom_pan":
                        # Zoom and pan effect
                        video_stream = (
                            ffmpeg
                            .input(image_path, loop=1, t=duration)
                            .filter('scale', self.width * 1.2, self.height * 1.2)
                            .filter('crop', self.width, self.height, f't*{int(duration*10)}', f't*{int(duration*5)}')
                        )
                    elif motion_type == "zoom_in":
                        # Zoom in effect
                        zoom_frames = int(duration * self.fps)
                        video_stream = (
                            ffmpeg
                            .input(image_path, loop=1, t=duration)
                            .filter('scale', self.width, self.height)
                            .filter('zoompan', z='min(zoom+0.002,1.3)', d=zoom_frames, x='iw/2-(iw/zoom/2)', y='ih/2-(ih/zoom/2)')
                        )
                    elif motion_type == "rotate":
                        # Rotation effect
                        video_stream = (
                            ffmpeg
                            .input(image_path, loop=1, t=duration)
                            .filter('scale', self.width, self.height)
                            .filter('rotate', angle=f't*{360/duration}', fillcolor='black')
                        )
                    elif motion_type == "scale":
                        # Scale effect
                        video_stream = (
                            ffmpeg
                            .input(image_path, loop=1, t=duration)
                            .filter('scale', self.width, self.height)
                            .filter('scale', f'iw*{1.1 + 0.1*sin(t)}', f'ih*{1.1 + 0.1*sin(t)}')
                        )
                    else:
                        # Default: static image
                        video_stream = (
                            ffmpeg
                            .input(image_path, loop=1, t=duration)
                            .filter('scale', self.width, self.height)
                        )
                except Exception as e:
                    logger.warning(f"FFmpeg motion effect {motion_type} failed: {e}, using static image")
                    # Fallback to static image
                    video_stream = (
                        ffmpeg
                        .input(image_path, loop=1, t=duration)
                        .filter('scale', self.width, self.height)
                    )
                
                # Create subtitle stream
                subtitle_stream = (
                    ffmpeg
                    .input(subtitle_path, loop=1, t=duration)
                    .filter('scale', self.width, self.height)
                )
                
                # Overlay subtitle on video
                combined = ffmpeg.overlay(video_stream, subtitle_stream, x=0, y=0)
                video_inputs.append(combined)
                
                current_time += duration
            
            # Concatenate all video segments
            if len(video_inputs) > 1:
                video = ffmpeg.concat(*video_inputs, v=1, a=0)
            else:
                video = video_inputs[0]
            
            # Add audio
            logger.info(f"Audio path: {audio_path}")
            audio = ffmpeg.input(audio_path)
            
            # Output final video
            logger.info(f"Rendering final video with FFmpeg motion effects to: {output_path}")
            (
                ffmpeg.output(
                    video, audio, output_path,
                    vcodec='h264_nvenc',           # GPU encoder
                    acodec='aac',
                    pix_fmt='yuv420p',
                    r=self.fps,
                    video_bitrate='4M',            # You can tweak this
                    audio_bitrate='128k',
                    **{'preset': 'fast'}           # Fastest reliable preset for NVENC
                ).overwrite_output().run(quiet=True)
            )
            
            logger.info(f"Video with FFmpeg motion effects created successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Error combining images with FFmpeg motion: {e}")
            # Fallback to simple video creation
            self.create_simple_video(audio_path, output_path, audio_duration)
    
    def cleanup(self):
        """Clean up resources"""
        if self.sd_generator:
            self.sd_generator.cleanup() 