import os
import time
import logging
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import shotstack_sdk as shotstack
from shotstack_sdk.model.soundtrack import Soundtrack
from shotstack_sdk.model.image_asset import ImageAsset
from shotstack_sdk.api import edit_api
from shotstack_sdk.model.clip import Clip
from shotstack_sdk.model.track import Track
from shotstack_sdk.model.timeline import Timeline
from shotstack_sdk.model.output import Output
from shotstack_sdk.model.edit import Edit
from shotstack_sdk.model.title_asset import TitleAsset

logger = logging.getLogger(__name__)

class ShotstackVideoCreator:
    def __init__(self):
        self.api_key = os.getenv("SHOTSTACK_API_KEY")
        if not self.api_key:
            logger.error("SHOTSTACK_API_KEY environment variable is not set")
            raise ValueError("SHOTSTACK_API_KEY environment variable is not set")
        
        # Initialize Shotstack configuration
        self.host = "https://api.shotstack.io/stage"
        self.configuration = shotstack.Configuration(host=self.host)
        self.configuration.api_key['DeveloperKey'] = self.api_key
        
        logger.info("ShotstackVideoCreator initialized successfully")
    
    def create_video(self, audio_url: str, narration_lines: List[dict], output_path: str) -> str:
        """Create a video using Shotstack API with audio, images, and subtitles"""
        
        try:
            logger.info(f"Creating video with Shotstack API for {len(narration_lines)} narration lines")
            
            # Step 1: Process audio URL (could be local file or S3 URL)
            processed_audio_url = self._process_audio_url(audio_url)
            logger.info(f"Using audio URL: {processed_audio_url}")
            
            # Step 2: Create video timeline
            edit = self._create_edit(processed_audio_url, narration_lines)
            
            # Step 3: Submit render job
            render_id = self._submit_render(edit)
            logger.info(f"Submitted render job: {render_id}")
            
            # Step 4: Wait for render to complete and download
            final_path = self._wait_and_download(render_id, output_path)
            
            # Step 5: Upload to S3
            s3_url = self._upload_video_to_s3(final_path)
            
            logger.info(f"Video created successfully: {final_path}")
            if s3_url:
                logger.info(f"Video uploaded to S3: {s3_url}")
                return s3_url
            else:
                logger.warning("Failed to upload to S3, returning local path")
                return final_path
            
        except Exception as e:
            logger.error(f"Error creating video with Shotstack: {e}")
            raise
    
    def create_video_with_images(self, audio_url: str, narration_lines: List[dict], image_urls: List[str], output_path: str) -> str:
        """Create a video using Shotstack API with audio, images, and subtitles"""
        
        try:
            logger.info(f"Creating video with images using Shotstack API for {len(narration_lines)} narration lines")
            
            # Step 1: Process audio URL
            processed_audio_url = self._process_audio_url(audio_url)
            logger.info(f"Using audio URL: {processed_audio_url}")
            
            # Step 2: Create video timeline with images
            edit = self._create_edit_with_images(processed_audio_url, narration_lines, image_urls)
            
            # Step 3: Submit render job
            render_id = self._submit_render(edit)
            logger.info(f"Submitted render job: {render_id}")
            
            # Step 4: Wait for render to complete and download
            final_path = self._wait_and_download(render_id, output_path)
            
            # Step 5: Upload to S3
            s3_url = self._upload_video_to_s3(final_path)
            
            logger.info(f"Video with images created successfully: {final_path}")
            if s3_url:
                logger.info(f"Video uploaded to S3: {s3_url}")
                return s3_url
            else:
                logger.warning("Failed to upload to S3, returning local path")
                return final_path
            
        except Exception as e:
            logger.error(f"Error creating video with images: {e}")
            raise
    
    def _process_audio_url(self, audio_input: str) -> str:
        """
        Process audio input which could be a local file path or S3 URL
        Returns a URL that Shotstack can access
        """
        # If it's already a URL (S3, HTTP, etc.), return as is
        if audio_input.startswith(('http://', 'https://')):
            logger.info(f"Using existing URL: {audio_input}")
            return audio_input
        
        # If it's a local file path, convert to file:// URL
        if os.path.exists(audio_input):
            abs_path = os.path.abspath(audio_input)
            file_url = f"file://{abs_path}"
            logger.info(f"Converted local file to URL: {file_url}")
            return file_url
        
        # If it doesn't exist, log error and use fallback
        logger.error(f"Audio file not found: {audio_input}")
        logger.info("Using fallback audio URL")
        return 'https://s3-ap-southeast-2.amazonaws.com/shotstack-assets/music/moment.mp3'
    
    def _upload_audio(self, audio_path: str) -> str:
        """Upload audio file to Shotstack and return the URL"""
        # This method is kept for backward compatibility
        # It now delegates to _process_audio_url
        return self._process_audio_url(audio_path)
    
    def _create_edit(self, audio_url: str, narration_lines: List[dict]) -> Edit:
        """Create the video edit with audio, background, and text overlays"""
        
        # Get total duration from audio or calculate from narration lines
        total_duration = sum(line.get("duration", 3.0) for line in narration_lines)
        
        # Create soundtrack
        soundtrack = Soundtrack(
            src=audio_url,
            effect="fadeInFadeOut"
        )
        
        # Create text overlays for each narration line
        text_clips = []
        current_time = 0.0
        
        for i, line in enumerate(narration_lines):
            duration = line.get("duration", 3.0)
            text = line["text"]
            
            # Create title asset for text
            title_asset = TitleAsset(
                style="minimal",
                text=text,
                size="medium"
            )
            
            # Create text clip
            text_clip = Clip(
                asset=title_asset,
                start=current_time,
                length=duration,
                effect="zoomIn"
            )
            
            text_clips.append(text_clip)
            current_time += duration
        
        # Create track for text overlays
        text_track = Track(clips=text_clips)
        
        # Create timeline with background color
        timeline = Timeline(
            background="#2C3E50",  # Dark blue-gray background
            soundtrack=soundtrack,
            tracks=[text_track]
        )
        
        # Create output
        output = Output(
            format="mp4",
            resolution="sd"
        )
        
        # Create edit
        edit = Edit(
            timeline=timeline,
            output=output
        )
        
        return edit
    
    def _create_edit_with_images(self, audio_url: str, narration_lines: List[dict], image_urls: List[str]) -> Edit:
        """Create the video edit with audio, images, and text overlays"""
        
        # Get total duration from audio or calculate from narration lines
        total_duration = sum(line.get("duration", 3.0) for line in narration_lines)
        
        # Create soundtrack
        soundtrack = Soundtrack(
            src=audio_url,
            effect="fadeInFadeOut"
        )
        
        # Create image and text clips for each narration line
        image_clips = []
        text_clips = []
        current_time = 0.0
        
        for i, line in enumerate(narration_lines):
            duration = line.get("duration", 3.0)
            text = line["text"]
            
            # Create image asset if available
            if i < len(image_urls) and image_urls[i]:
                image_asset = ImageAsset(
                    src=image_urls[i],
                    type="image"
                )
                
                # Create image clip
                image_clip = Clip(
                    asset=image_asset,
                    start=current_time,
                    length=duration,
                    effect="zoomIn"
                )
                image_clips.append(image_clip)
            
            # Create title asset for text
            title_asset = TitleAsset(
                style="minimal",
                text=text,
                size="medium"
            )
            
            # Create text clip
            text_clip = Clip(
                asset=title_asset,
                start=current_time,
                length=duration,
                effect="zoomIn"
            )
            
            text_clips.append(text_clip)
            current_time += duration
        
        # Create tracks
        image_track = Track(clips=image_clips)
        text_track = Track(clips=text_clips)
        
        # Create timeline with background color
        timeline = Timeline(
            background="#2C3E50",  # Dark blue-gray background
            soundtrack=soundtrack,
            tracks=[image_track, text_track]  # Images on bottom track, text on top
        )
        
        # Create output
        output = Output(
            format="mp4",
            resolution="sd"
        )
        
        # Create edit
        edit = Edit(
            timeline=timeline,
            output=output
        )
        
        return edit
    
    def _submit_render(self, edit: Edit) -> str:
        """Submit a render job to Shotstack"""
        with shotstack.ApiClient(self.configuration) as api_client:
            api_instance = edit_api.EditApi(api_client)
            
            try:
                api_response = api_instance.post_render(edit)
                render_id = api_response['response']['id']
                logger.info(f"Render submitted successfully: {render_id}")
                return render_id
            except Exception as e:
                logger.error(f"Error submitting render: {e}")
                raise
    
    def _wait_and_download(self, render_id: str, output_path: str) -> str:
        """Wait for render to complete and download the video"""
        with shotstack.ApiClient(self.configuration) as api_client:
            api_instance = edit_api.EditApi(api_client)
            
            # Poll for completion
            while True:
                try:
                    api_response = api_instance.get_render(render_id)
                    status = api_response['response']['status']
                    logger.info(f"Render status: {status}")
                    
                    if status == "done":
                        # Download the video
                        download_url = api_response['response']['url']
                        self._download_video(download_url, output_path)
                        return output_path
                    elif status == "failed":
                        error = api_response['response'].get('error', 'Unknown error')
                        raise Exception(f"Video rendering failed: {error}")
                    
                    time.sleep(5)  # Wait 5 seconds before checking again
                    
                except Exception as e:
                    logger.error(f"Error checking render status: {e}")
                    raise
    
    def _download_video(self, download_url: str, output_path: str):
        """Download the rendered video"""
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Video downloaded to: {output_path}")
    
    def _upload_video_to_s3(self, video_path: str) -> Optional[str]:
        """
        Upload the rendered video to S3 under youtube-shorts folder
        
        Args:
            video_path: Local path to the video file
            
        Returns:
            S3 URL of the uploaded video, or None if upload failed
        """
        try:
            # Import S3Uploader here to avoid circular imports
            from .s3_uploader import S3Uploader
            
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return None
            
            # Initialize S3 uploader
            s3_uploader = S3Uploader()
            
            # Generate S3 key for youtube-shorts folder
            import uuid
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = os.path.basename(video_path)
            s3_key = f"youtube-shorts/{timestamp}_{unique_id}_{filename}"
            
            # Upload to S3
            s3_url = s3_uploader.upload_file_with_custom_key(
                video_path,
                s3_key,
                "video/mp4"
            )
            
            if s3_url:
                logger.info(f"Video uploaded to S3: {s3_url}")
                return s3_url
            else:
                logger.error("Failed to upload video to S3")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading video to S3: {e}")
            return None
    
    def create_simple_video(self, audio_path: str, output_path: str, duration: float = 30.0) -> str:
        """Create a simple video with just audio and a static background"""
        try:
            logger.info("Creating simple video with Shotstack")
            
            # Upload audio
            audio_url = self._upload_audio(audio_path)
            
            # Create soundtrack
            soundtrack = Soundtrack(
                src=audio_url,
                effect="fadeInFadeOut"
            )
            
            # Create title
            title_asset = TitleAsset(
                style="minimal",
                text="Why Would You",
                size="medium"
            )
            
            title_clip = Clip(
                asset=title_asset,
                start=0.0,
                length=duration,
                effect="zoomIn"
            )
            
            # Create track
            title_track = Track(clips=[title_clip])
            
            # Create timeline
            timeline = Timeline(
                background="#34495E",
                soundtrack=soundtrack,
                tracks=[title_track]
            )
            
            # Create output
            output = Output(
                format="mp4",
                resolution="sd"
            )
            
            # Create edit
            edit = Edit(
                timeline=timeline,
                output=output
            )
            
            # Submit render
            render_id = self._submit_render(edit)
            final_path = self._wait_and_download(render_id, output_path)
            
            # Upload to S3
            s3_url = self._upload_video_to_s3(final_path)
            
            if s3_url:
                logger.info(f"Simple video uploaded to S3: {s3_url}")
                return s3_url
            else:
                logger.warning("Failed to upload simple video to S3, returning local path")
                return final_path
            
        except Exception as e:
            logger.error(f"Error creating simple video: {e}")
            raise
    
    def get_render_status(self, render_id: str) -> Dict:
        """Get the status of a render job"""
        with shotstack.ApiClient(self.configuration) as api_client:
            api_instance = edit_api.EditApi(api_client)
            
            try:
                api_response = api_instance.get_render(render_id)
                return api_response['response']
            except Exception as e:
                logger.error(f"Error getting render status: {e}")
                raise
    
    def list_renders(self, limit: int = 10) -> List[Dict]:
        """List recent render jobs"""
        with shotstack.ApiClient(self.configuration) as api_client:
            api_instance = edit_api.EditApi(api_client)
            
            try:
                api_response = api_instance.get_renders(limit=limit)
                return api_response['response']['renders']
            except Exception as e:
                logger.error(f"Error listing renders: {e}")
                raise 