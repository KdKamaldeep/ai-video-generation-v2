import os
import tempfile
import uuid
import shutil
import logging
import requests
from typing import List, Dict
import ffmpeg

logger = logging.getLogger(__name__)

class FFmpegVideoCreator:
    def __init__(self):
        self.width = 1080
        self.height = 1920  # Full size vertical format for YouTube Shorts
        self.fps = 30
        self.video_codec = 'libx264'
        self.audio_codec = 'aac'
        # Animation types available
        self.animation_types = ['zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'static']
        
    def create_video_with_images(self, audio_path: str, narration_lines: List[Dict], output_path: str, animation_type: str = 'zoom_in') -> str:
        """
        Create a vertical YouTube Shorts video with images and text overlays.
        
        Args:
            audio_path: Path to MP3 or WAV audio file (local or S3 URL)
            narration_lines: List of dicts with 'text', 'duration', and 'visual_suggestion' (S3 image URL or local file path)
            output_path: Where to save the final video
            animation_type: Type of animation ('zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'static')
            
        Returns:
            Path to the created video file
        """
        logger.info(f"Creating video with {len(narration_lines)} narration lines")
        
        # Create temporary directory for all assets
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Download images and create video segments
            segment_paths = []
            
            for i, line in enumerate(narration_lines):
                logger.info(f"Processing narration line {i + 1}/{len(narration_lines)}")
                
                # Download image from S3 or use local path
                image_path = self._download_image(line['visual_suggestion'], temp_dir, f"image_{i}")
                
                # Create video segment with text overlay and animation
                segment_path = os.path.join(temp_dir, f"segment_{i}.mp4")
                
                # Try simple version first to avoid zoompan issues
                try:
                    self._create_video_segment_simple(
                        image_path=image_path,
                        text=line['text'],
                        duration=line['duration'],
                        output_path=segment_path
                    )
                except Exception as e:
                    logger.warning(f"Simple segment creation failed, trying with animation: {e}")
                    self._create_video_segment(
                        image_path=image_path,
                        text=line['text'],
                        duration=line['duration'],
                        output_path=segment_path,
                        animation_type=animation_type
                    )
                
                segment_paths.append(segment_path)
            
            # Concatenate all segments
            concatenated_path = os.path.join(temp_dir, "concatenated.mp4")
            self._concatenate_segments(segment_paths, concatenated_path)
            
            # Add audio to final video
            self._add_audio_to_video(concatenated_path, audio_path, output_path)
            
            logger.info(f"Video created successfully: {output_path}")
            return output_path
            
        finally:
            # Clean up temporary files
            self._cleanup_temp_files(temp_dir)
    
    def test_ffmpeg_installation(self):
        """
        Test if FFmpeg is properly installed and working.
        """
        try:
            logger.info("Testing FFmpeg installation...")
            
            # Test basic FFmpeg command
            (
                ffmpeg
                .input('color=c=red:size=100x100:duration=1', f='lavfi')
                .output('test_output.mp4', vcodec='libx264', pix_fmt='yuv420p')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            logger.info("FFmpeg test successful")
            return True
            
        except Exception as e:
            logger.error(f"FFmpeg test failed: {e}")
            return False
    
    def create_simple_test_video(self, output_path: str = "test_video.mp4"):
        """
        Create a simple test video to verify FFmpeg functionality.
        """
        try:
            logger.info("Creating simple test video...")
            
            (
                ffmpeg
                .input('color=c=blue:size=1080x1920:duration=3', f='lavfi')
                .filter('drawtext', text='Test Video', fontsize=60, fontcolor='white', x='(w-text_w)/2', y='(h-text_h)/2')
                .output(
                    output_path,
                    vcodec='libx264',
                    acodec='none',
                    r=30,
                    pix_fmt='yuv420p',
                    s='1080x1920'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            logger.info(f"Test video created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create test video: {e}")
            raise
    
    def _download_image(self, image_source: str, temp_dir: str, filename: str) -> str:
        """
        Download image from S3 URL or use local file path.
        
        Args:
            image_source: S3 URL or local file path of the image
            temp_dir: Temporary directory to save the image (if downloading)
            filename: Base filename for the image
            
        Returns:
            Path to the image (local file path)
        """
        logger.info(f"Processing image source: {image_source}")
        
        # Check if it's a local file path
        if os.path.exists(image_source):
            logger.info(f"Using local image: {image_source}")
            return image_source
        
        # Assume it's an S3 URL and try to download
        try:
            # Determine file extension from URL or default to .jpg
            if '.' in image_source.split('/')[-1]:
                ext = '.' + image_source.split('.')[-1].split('?')[0]
            else:
                ext = '.jpg'
            
            image_path = os.path.join(temp_dir, f"{filename}{ext}")
            
            logger.info(f"Downloading image from {image_source}")
            logger.info(f"Will save to: {image_path}")
            
            response = requests.get(image_source, timeout=30)
            response.raise_for_status()
            
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            # Verify the file was created and has content
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Failed to create image file: {image_path}")
            
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                raise ValueError(f"Downloaded image file is empty: {image_path}")
            
            logger.info(f"Image downloaded to {image_path} (size: {file_size} bytes)")
            return image_path
            
        except Exception as e:
            logger.error(f"Failed to download image from {image_source}: {e}")
            raise
    

    
    def _create_video_segment(self, image_path: str, text: str, duration: float, output_path: str, animation_type: str = 'zoom_in'):
        """
        Create a video segment with image and text overlay with animation.
        
        Args:
            image_path: Path to the background image
            text: Text to overlay on the video
            duration: Duration of the segment in seconds
            output_path: Where to save the segment
            animation_type: Type of animation to apply
        """
        try:
            # Verify image file exists
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}, using fallback colored background")
                return self._create_fallback_segment(text, duration, output_path, animation_type)
            
            logger.info(f"Creating video segment for image: {image_path}")
            logger.info(f"Text: {text[:50]}...")
            logger.info(f"Duration: {duration}s, Animation: {animation_type}")
            
            # Create text overlay filter
            text_filter = self._create_positioned_text_filter(text, 'bottom')
            
            # Create animation filter
            logger.info("Creating animation filter...")
            animation_filter = self._create_animation_filter(animation_type, duration)
            
            logger.info("Building FFmpeg command...")
            # Create video from image with text overlay and animation
            (
                ffmpeg
                .input(image_path, loop=1, t=duration)
                .filter('scale', self.width, self.height, force_original_aspect_ratio='decrease')
                .filter('pad', self.width, self.height, '(ow-iw)/2', '(oh-ih)/2', color='black')
                .filter('zoompan', **animation_filter)
                .filter('drawtext', **text_filter)
                .output(
                    output_path,
                    vcodec=self.video_codec,
                    acodec='none',  # No audio in segments
                    r=self.fps,
                    pix_fmt='yuv420p',
                    s=f'{self.width}x{self.height}'  # Explicitly set output resolution
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            logger.info(f"Created video segment with {animation_type} animation: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create video segment: {e}")
            logger.error(f"Image path: {image_path}")
            logger.error(f"Text: {text}")
            logger.error(f"Duration: {duration}, Animation: {animation_type}")
            logger.info("Trying fallback segment creation...")
            return self._create_fallback_segment(text, duration, output_path, animation_type)
    
    def _create_video_segment_simple(self, image_path: str, text: str, duration: float, output_path: str):
        """
        Create a video segment without complex animations to test if zoompan is causing issues.
        """
        try:
            logger.info(f"Creating simple video segment for image: {image_path}")
            
            # Verify image file exists
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}, using fallback colored background")
                return self._create_fallback_segment(text, duration, output_path, 'static')
            
            # Create text overlay filter
            text_filter = self._create_positioned_text_filter(text, 'bottom')
            
            logger.info("Building simple FFmpeg command (no zoompan)...")
            # Create video from image with text overlay but no animation
            (
                ffmpeg
                .input(image_path, loop=1, t=duration)
                .filter('scale', self.width, self.height, force_original_aspect_ratio='decrease')
                .filter('pad', self.width, self.height, '(ow-iw)/2', '(oh-ih)/2', color='black')
                #.filter('drawtext', **text_filter)
                .output(
                    output_path,
                    vcodec=self.video_codec,
                    acodec='none',  # No audio in segments
                    r=self.fps,
                    pix_fmt='yuv420p',
                    s=f'{self.width}x{self.height}'  # Explicitly set output resolution
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            logger.info(f"Created simple video segment: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create simple video segment: {e}")
            logger.error(f"Image path: {image_path}")
            logger.error(f"Text: {text}")
            logger.error(f"Duration: {duration}")
            logger.info("Trying fallback segment creation...")
            return self._create_fallback_segment(text, duration, output_path, 'static')
    
    def _create_fallback_segment(self, text: str, duration: float, output_path: str, animation_type: str = 'static'):
        """
        Create a fallback video segment with colored background when image processing fails.
        """
        try:
            logger.info("Creating fallback segment with colored background")
            
            # Create text overlay filter
            text_filter = self._create_positioned_text_filter(text, 'bottom')
            
            # Create a simple colored background video
            (
                ffmpeg
                .input('color=c=0x2C3E50:size=1080x1920:duration=' + str(duration), f='lavfi')
                .filter('drawtext', **text_filter)
                .output(
                    output_path,
                    vcodec=self.video_codec,
                    acodec='none',
                    r=self.fps,
                    pix_fmt='yuv420p',
                    s=f'{self.width}x{self.height}'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            logger.info(f"Created fallback video segment: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create fallback segment: {e}")
            logger.info("Trying ultra-simple fallback...")
            return self._create_ultra_simple_segment(text, duration, output_path)
    
    def _create_ultra_simple_segment(self, text: str, duration: float, output_path: str):
        """
        Create an ultra-simple video segment with just a colored background and basic text.
        """
        try:
            logger.info("Creating ultra-simple segment")
            
            # Create a very basic colored background video without text overlay
            (
                ffmpeg
                .input('color=c=0x2C3E50:size=1080x1920:duration=' + str(duration), f='lavfi')
                .output(
                    output_path,
                    vcodec=self.video_codec,
                    acodec='none',
                    r=self.fps,
                    pix_fmt='yuv420p',
                    s=f'{self.width}x{self.height}'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            logger.info(f"Created ultra-simple video segment: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create ultra-simple segment: {e}")
            raise
    
    def _create_animation_filter(self, animation_type: str, duration: float) -> Dict:
        """
        Create animation filter parameters for ffmpeg zoompan filter.
        
        Args:
            animation_type: Type of animation ('zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'static')
            duration: Duration of the segment in seconds
            
        Returns:
            Dictionary of animation filter parameters
        """
        fps = self.fps
        total_frames = int(duration * fps)
        
        if animation_type == 'zoom_in':
            return {
                'z': f'min(zoom+0.0015,1.5)',
                'd': total_frames,
                's': f'{self.width}x{self.height}',
                'x': 'iw/2-(iw/zoom/2)',
                'y': 'ih/2-(ih/zoom/2)'
            }
        elif animation_type == 'zoom_out':
            return {
                'z': f'if(lte(zoom,1.0),1.5,max(1.001,zoom-0.0015))',
                'd': total_frames,
                's': f'{self.width}x{self.height}',
                'x': 'iw/2-(iw/zoom/2)',
                'y': 'ih/2-(ih/zoom/2)'
            }
        elif animation_type == 'pan_left':
            return {
                'z': '1.2',
                'd': total_frames,
                's': f'{self.width}x{self.height}',
                'x': f'if(lte(on,{total_frames//2}),iw/2-(iw/zoom/2),iw/2-(iw/zoom/2)-{self.width//4})',
                'y': 'ih/2-(ih/zoom/2)'
            }
        elif animation_type == 'pan_right':
            return {
                'z': '1.2',
                'd': total_frames,
                's': f'{self.width}x{self.height}',
                'x': f'if(lte(on,{total_frames//2}),iw/2-(iw/zoom/2),iw/2-(iw/zoom/2)+{self.width//4})',
                'y': 'ih/2-(ih/zoom/2)'
            }
        else:  # static
            return {
                'z': '1',
                'd': total_frames,
                's': f'{self.width}x{self.height}',
                'x': 'iw/2-(iw/zoom/2)',
                'y': 'ih/2-(ih/zoom/2)'
            }
    
    def _create_text_filter(self, text: str) -> Dict:
        """
        Create text filter parameters for ffmpeg drawtext filter.
        
        Args:
            text: Text to display
            
        Returns:
            Dictionary of text filter parameters
        """
        logger.info(f"Creating text filter for text: {text}")
        
        # Escape special characters in text for ffmpeg
        # Replace single quotes with escaped single quotes
        escaped_text = text.replace("'", "\\'").replace('"', '\\"')
        
        # Additional escaping for ffmpeg drawtext filter
        escaped_text = escaped_text.replace(':', '\\:').replace('[', '\\[').replace(']', '\\]')
        
        return {
            'text': escaped_text,
            'fontsize': 60,
            'fontcolor': 'white',
            'x': '(w-text_w)/2',  # Center horizontally
            'y': 'h-text_h-100',  # Position text near bottom with margin
            'shadowcolor': 'black',
            'shadowx': 5,
            'shadowy': 5,
            'box': 1,
            'boxcolor': 'black@0.8',  # More opaque background for better readability
            'boxborderw': 12,
            'line_spacing': 15
        }
    
    def _create_smart_text_filter(self, text: str) -> Dict:
        """
        Create text filter with smart positioning to avoid overlapping with main subject.
        
        Args:
            text: Text to display
            
        Returns:
            Dictionary of text filter parameters
        """
        logger.info(f"Creating smart text filter for text: {text}")
        
        # Escape special characters in text for ffmpeg
        escaped_text = text.replace("'", "\\'").replace('"', '\\"')
        escaped_text = escaped_text.replace(':', '\\:').replace('[', '\\[').replace(']', '\\]')
        
        # Split text into lines if it's too long
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= 20:  # Shorter line length for better fit
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Join lines with newline character
        multiline_text = "\\n".join(lines)
        
        return {
            'text': multiline_text,
            'fontsize': 60,  # Slightly smaller font
            'fontcolor': 'white',
            'x': '(w-text_w)/2',  # Center horizontally
            'y': 'h-text_h-120',  # Position text near bottom with more margin
            'shadowcolor': 'black',
            'shadowx': 6,
            'shadowy': 6,
            'box': 1,
            'boxcolor': 'black@0.9',  # Very opaque background
            'boxborderw': 15,
            'line_spacing': 20
        }
    
    def _create_positioned_text_filter(self, text: str, position: str = 'bottom') -> Dict:
        """
        Create text filter with specific positioning options.
        
        Args:
            text: Text to display
            position: Position for text ('bottom', 'top', 'center')
            
        Returns:
            Dictionary of text filter parameters
        """
        logger.info(f"Creating positioned text filter for text: {text} at {position}")
        
        # Escape special characters in text for ffmpeg
        escaped_text = text.replace("'", "\\'").replace('"', '\\"')
        escaped_text = escaped_text.replace(':', '\\:').replace('[', '\\[').replace(']', '\\]')
        
        # Split text into lines if it's too long
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= 18:  # Even shorter for better fit
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Join lines with newline character
        multiline_text = "\\n".join(lines)
        
        # Set positioning based on preference
        if position == 'bottom':
            y_position = 'h-text_h-150'  # Further from bottom
        elif position == 'top':
            y_position = 'text_h+50'  # Near top
        else:  # center
            y_position = '(h-text_h)/2'  # Center
        
        return {
            'text': multiline_text,
            'fontsize': 55,  # Smaller font for better fit
            'fontcolor': 'white',
            'x': '(w-text_w)/2',  # Center horizontally
            'y': y_position,
            'shadowcolor': 'black',
            'shadowx': 7,
            'shadowy': 7,
            'box': 1,
            'boxcolor': 'black@0.95',  # Very opaque background
            'boxborderw': 18,
            'line_spacing': 25
        }
    
    def _concatenate_segments(self, segment_paths: List[str], output_path: str):
        """
        Concatenate multiple video segments into one video.
        
        Args:
            segment_paths: List of paths to video segments
            output_path: Where to save the concatenated video
        """
        try:
            # Create a temporary file with segment paths
            temp_list_path = os.path.join(os.path.dirname(output_path), 'segments.txt')
            
            with open(temp_list_path, 'w') as f:
                for segment_path in segment_paths:
                    f.write(f"file '{segment_path}'\n")
            
            # Concatenate segments
            (
                ffmpeg
                .input(temp_list_path, f='concat', safe=0)
                .output(
                    output_path,
                    vcodec=self.video_codec,
                    acodec='none',
                    r=self.fps,
                    pix_fmt='yuv420p',
                    s=f'{self.width}x{self.height}'  # Explicitly set output resolution
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            # Clean up temp list file
            os.remove(temp_list_path)
            
            logger.info(f"Concatenated {len(segment_paths)} segments to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to concatenate segments: {e}")
            raise
    
    def _add_audio_to_video(self, video_path: str, audio_path: str, output_path: str):
        """
        Add audio to the video, matching the total duration.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to the audio file (local or S3 URL)
            output_path: Where to save the final video with audio
        """
        try:
            # If audio_path is an S3 URL, download it first
            local_audio_path = audio_path
            temp_audio_path = None
            
            if audio_path.startswith('http'):
                temp_audio_path = os.path.join(os.path.dirname(video_path), f"temp_audio_{uuid.uuid4().hex[:8]}.mp3")
                self._download_audio(audio_path, temp_audio_path)
                local_audio_path = temp_audio_path
            
            # Add audio to video using ffmpeg
            video_input = ffmpeg.input(video_path)
            audio_input = ffmpeg.input(local_audio_path)
            
            (
                ffmpeg
                .output(
                    video_input,
                    audio_input,
                    output_path,
                    vcodec=self.video_codec,
                    acodec=self.audio_codec,
                    r=self.fps,
                    pix_fmt='yuv420p',
                    s=f'{self.width}x{self.height}',  # Explicitly set output resolution
                    shortest=None  # End when shortest input ends
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            # Clean up temporary audio file if created
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            logger.info(f"Added audio to video: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to add audio to video: {e}")
            raise
    
    def _download_audio(self, audio_url: str, output_path: str):
        """
        Download audio file from URL.
        
        Args:
            audio_url: URL of the audio file
            output_path: Where to save the audio file
        """
        try:
            logger.info(f"Downloading audio from {audio_url}")
            response = requests.get(audio_url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Audio downloaded to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to download audio from {audio_url}: {e}")
            raise
    
    def _cleanup_temp_files(self, temp_dir: str):
        """
        Clean up all temporary files and directories.
        
        Args:
            temp_dir: Path to the temporary directory to clean up
        """
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to clean up temporary files: {e}") 