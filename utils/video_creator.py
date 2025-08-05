import os
import ffmpeg
import tempfile
from typing import List
from PIL import Image, ImageDraw, ImageFont
import json
import logging

logger = logging.getLogger(__name__)

class VideoCreator:
    def __init__(self):
        self.width = 1080
        self.height = 1920  # 9:16 aspect ratio for YouTube Shorts
        self.fps = 30
    
    def create_video(self, audio_path: str, narration_lines: List[dict], output_path: str) -> str:
        """Create a vertical video with audio, images, and subtitles"""
        
        logger.info(f"Creating video with {len(narration_lines)} narration lines")
        
        # Create temporary directory for assets
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate placeholder images for each narration line
            image_paths = []
            subtitle_paths = []
            
            for i, line in enumerate(narration_lines):
                # Create placeholder image
                img_path = os.path.join(temp_dir, f"image_{i}.png")
                self._create_placeholder_image(img_path, line.get("visual_suggestion", "placeholder"))
                image_paths.append(img_path)
                
                # Create subtitle image
                subtitle_path = os.path.join(temp_dir, f"subtitle_{i}.png")
                self._create_subtitle_image(subtitle_path, line["text"])
                subtitle_paths.append(subtitle_path)
            
            logger.info(f"Generated {len(image_paths)} images and {len(subtitle_paths)} subtitles")
            
            # Create video using FFmpeg
            self._combine_assets(audio_path, image_paths, subtitle_paths, narration_lines, output_path)
            
        return output_path
    
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
    
    def _combine_assets(self, audio_path: str, image_paths: List[str], 
                       subtitle_paths: List[str], narration_lines: List[dict], output_path: str):
        """Combine audio, images, and subtitles into final video"""
        
        try:
            # Get audio duration
            probe = ffmpeg.probe(audio_path)
            audio_duration = float(probe['streams'][0]['duration'])
            logger.info(f"Audio duration: {audio_duration} seconds")
            
            # Create video stream from images
            video_inputs = []
            current_time = 0
            
            for i, (image_path, subtitle_path, line) in enumerate(zip(image_paths, subtitle_paths, narration_lines)):
                duration = line.get("duration", 3.0)
                logger.info(f"Processing segment {i+1}: duration={duration}s")
                
                # Create image stream
                image_stream = (
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
                
                # Overlay subtitle on image
                combined = ffmpeg.overlay(image_stream, subtitle_stream, x=0, y=0)
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
                (
                    ffmpeg
                    .input(tmp_img.name, loop=1, t=duration)
                    .input(audio_path)
                    .output(output_path,
                           vcodec='libx264', acodec='aac',
                           pix_fmt='yuv420p', r=self.fps,
                           video_bitrate='2M', audio_bitrate='128k')
                    .overwrite_output()
                    .run(quiet=True)
                )
                
                # Clean up
                os.unlink(tmp_img.name)
            
            logger.info(f"Simple video created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating simple video: {e}")
            raise 