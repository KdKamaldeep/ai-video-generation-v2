import os
import ffmpeg
import tempfile
from typing import List
from PIL import Image, ImageDraw, ImageFont
import json

class VideoCreator:
    def __init__(self):
        self.width = 1080
        self.height = 1920  # 9:16 aspect ratio for YouTube Shorts
        self.fps = 30
    
    def create_video(self, audio_path: str, narration_lines: List[dict], output_path: str) -> str:
        """Create a vertical video with audio, images, and subtitles"""
        
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
            
            # Create video using FFmpeg
            self._combine_assets(audio_path, image_paths, subtitle_paths, narration_lines, output_path)
            
        return output_path
    
    def _create_placeholder_image(self, output_path: str, description: str):
        """Create a placeholder image with description"""
        img = Image.new('RGB', (self.width, self.height), color='#2C3E50')
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except:
            font = ImageFont.load_default()
        
        # Add description text
        text = f"Image: {description}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (self.width - text_width) // 2
        y = (self.height - text_height) // 2
        
        draw.text((x, y), text, fill='white', font=font)
        img.save(output_path)
    
    def _create_subtitle_image(self, output_path: str, text: str):
        """Create a subtitle image with text"""
        img = Image.new('RGBA', (self.width, 200), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
        
        # Add background rectangle
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (self.width - text_width) // 2
        y = (200 - text_height) // 2
        
        # Draw background
        padding = 20
        draw.rectangle([
            x - padding, y - padding,
            x + text_width + padding, y + text_height + padding
        ], fill=(0, 0, 0, 180))
        
        # Draw text
        draw.text((x, y), text, fill='white', font=font)
        img.save(output_path)
    
    def _combine_assets(self, audio_path: str, image_paths: List[str], 
                       subtitle_paths: List[str], narration_lines: List[dict], output_path: str):
        """Combine audio, images, and subtitles into final video"""
        
        # Get audio duration
        probe = ffmpeg.probe(audio_path)
        audio_duration = float(probe['streams'][0]['duration'])
        
        # Create video stream from images
        video_inputs = []
        current_time = 0
        
        for i, (image_path, subtitle_path, line) in enumerate(zip(image_paths, subtitle_paths, narration_lines)):
            duration = line.get("duration", 3.0)
            
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
                .filter('scale', self.width, 200)
                .filter('pad', self.width, self.height, 0, self.height - 200)
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
        (
            ffmpeg
            .output(video, audio, output_path,
                   vcodec='libx264', acodec='aac',
                   pix_fmt='yuv420p', r=self.fps,
                   video_bitrate='2M', audio_bitrate='128k')
            .overwrite_output()
            .run(quiet=True)
        )
    
    def create_simple_video(self, audio_path: str, output_path: str, duration: float = 30.0) -> str:
        """Create a simple video with just audio and a static background"""
        
        # Create a simple background image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
            img = Image.new('RGB', (self.width, self.height), color='#34495E')
            draw = ImageDraw.Draw(img)
            
            # Add channel name
            try:
                font = ImageFont.truetype("arial.ttf", 80)
            except:
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
        
        return output_path 