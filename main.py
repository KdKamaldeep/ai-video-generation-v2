import os
import tempfile
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import time

# Import utility modules
from utils.script_generator import ScriptGenerator, GeneratedScript, ScriptLine
from utils.voice_synthesizer import VoiceSynthesizer
from utils.shotstack_video_creator import ShotstackVideoCreator
from utils.youtube_uploader import YouTubeUploader
from utils.s3_uploader import S3Uploader
from utils.image_generator import ImageGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Why Would You - YouTube Shorts Automation",
    description="Automated YouTube Shorts generation for the 'Why Would You' channel",
    version="1.0.0"
)

# Initialize utility classes
logger.info("Initializing utility classes...")

try:
    script_generator = ScriptGenerator()
    logger.info("ScriptGenerator initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize ScriptGenerator: {e}")
    script_generator = None

try:
    voice_synthesizer = VoiceSynthesizer()
    logger.info("VoiceSynthesizer initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize VoiceSynthesizer: {e}")
    voice_synthesizer = None

try:
    video_creator = ShotstackVideoCreator()
    logger.info("ShotstackVideoCreator initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize ShotstackVideoCreator: {e}")
    video_creator = None

try:
    youtube_uploader = YouTubeUploader()
    logger.info("YouTubeUploader initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize YouTubeUploader: {e}")
    youtube_uploader = None

try:
    s3_uploader = S3Uploader()
    logger.info("S3Uploader initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize S3Uploader: {e}")
    s3_uploader = None

try:
    image_generator = ImageGenerator()
    logger.info("ImageGenerator initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize ImageGenerator: {e}")
    image_generator = None

# Pydantic models for API requests/responses
class VoiceRequest(BaseModel):
    narration: List[str]

class VideoRequest(BaseModel):
    audio_path: str
    narration_lines: List[dict]

class UploadRequest(BaseModel):
    video_path: str
    title: str
    description: str
    tags: List[str]

class S3VoiceRequest(BaseModel):
    narration: List[str]

class S3VideoRequest(BaseModel):
    audio_url: str
    narration_lines: List[dict]

class ImageVideoRequest(BaseModel):
    audio_url: str
    narration_lines: List[dict]
    image_urls: List[str]

class GenerateImagesRequest(BaseModel):
    script_lines: List[str]
    style: str = "realistic"

# Create necessary directories
logger.info("Creating necessary directories...")
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)
logger.info("Directories created successfully")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    logger.info("Root endpoint accessed")
    return {
        "message": "Why Would You - YouTube Shorts Automation API",
        "version": "1.0.0",
        "endpoints": {
            "generate_script": "/generate-script",
            "generate_voice": "/generate-voice",
            "create_video": "/create-video",
            "upload_to_youtube": "/upload-to-youtube",
            "full_pipeline": "/full-pipeline",
            "generate_voice_s3": "/generate-voice-s3",
            "create_video_s3": "/create-video-s3",
            "full_pipeline_s3": "/full-pipeline-s3",
            "generate_images": "/generate-images",
            "create_video_with_images": "/create-video-with-images",
            "full_pipeline_with_images": "/full-pipeline-with-images",
            "logs": "/logs",
            "docs": "/docs"
        },
        "s3_integration": {
            "enabled": s3_uploader is not None,
            "bucket": os.getenv("S3_BUCKET_NAME", "why-would-you"),
            "description": "S3 integration allows uploading audio files and images to AWS S3 and using S3 URLs with Shotstack for video creation"
        },
        "image_generation": {
            "enabled": image_generator is not None,
            "model": "DALL-E 3",
            "description": "AI-powered image generation using DALL-E 3, uploaded to S3 thumbs folder"
        }
    }

@app.get("/generate-script")
async def generate_script():
    """Generate a YouTube Shorts script using OpenAI GPT"""
    logger.info("Script generation requested")
    
    if script_generator is None:
        logger.error("Script generator not available")
        raise HTTPException(status_code=503, detail="Script generator not available. Check OPENAI_API_KEY configuration.")
    
    try:
        logger.info("Generating script using OpenAI...")
        script = script_generator.generate_script()
        logger.info(f"Script generated successfully: {script.title}")
        return {
            "success": True,
            "script": {
                "title": script.title,
                "narration": [line.dict() for line in script.narration],
                "total_duration": script.total_duration,
                "tags": script.tags
            }
        }
    except Exception as e:
        logger.error(f"Failed to generate script: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate script: {str(e)}")

@app.post("/generate-voice")
async def generate_voice(request: VoiceRequest):
    """Generate voiceover using ElevenLabs API"""
    logger.info(f"Voice generation requested for {len(request.narration)} lines")
    
    if voice_synthesizer is None:
        logger.error("Voice synthesizer not available")
        raise HTTPException(status_code=503, detail="Voice synthesizer not available. Check ELEVENLABS_API_KEY configuration.")
    
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir="output") as tmp_file:
            audio_path = tmp_file.name
        
        logger.info(f"Generating voice using ElevenLabs...")
        # Generate voice
        result_path = voice_synthesizer.synthesize_voice(request.narration, audio_path)
        logger.info(f"Voice generated successfully: {result_path}")
        
        return {
            "success": True,
            "audio_path": result_path,
            "message": "Voice generated successfully"
        }
    except Exception as e:
        logger.error(f"Failed to generate voice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate voice: {str(e)}")

@app.post("/generate-voice-s3")
async def generate_voice_s3(request: S3VoiceRequest):
    """Generate voiceover using ElevenLabs API and upload to S3"""
    logger.info(f"S3 Voice generation requested for {len(request.narration)} lines")
    
    if voice_synthesizer is None:
        logger.error("Voice synthesizer not available")
        raise HTTPException(status_code=503, detail="Voice synthesizer not available. Check ELEVENLABS_API_KEY configuration.")
    
    if s3_uploader is None:
        logger.error("S3 uploader not available")
        raise HTTPException(status_code=503, detail="S3 uploader not available. Check AWS credentials configuration.")
    
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir="output") as tmp_file:
            audio_path = tmp_file.name
        
        logger.info(f"Generating voice and uploading to S3...")
        # Generate voice and upload to S3
        s3_url = voice_synthesizer.synthesize_and_upload_to_s3(request.narration, audio_path)
        
        if s3_url:
            logger.info(f"Voice generated and uploaded to S3: {s3_url}")
            return {
                "success": True,
                "audio_url": s3_url,
                "local_path": audio_path,
                "message": "Voice generated and uploaded to S3 successfully"
            }
        else:
            logger.error("Failed to generate voice or upload to S3")
            raise HTTPException(status_code=500, detail="Failed to generate voice or upload to S3")
            
    except Exception as e:
        logger.error(f"Failed to generate voice with S3: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate voice with S3: {str(e)}")

@app.post("/create-video-s3")
async def create_video_s3(request: S3VideoRequest):
    """Create video using Shotstack with S3 audio URL"""
    logger.info(f"S3 Video creation requested for audio URL: {request.audio_url}")
    
    if video_creator is None:
        logger.error("Video creator not available")
        raise HTTPException(status_code=503, detail="Video creator not available.")
    
    try:
        # Create output video path with timestamp
        timestamp = int(time.time())
        video_filename = f"shorts_{timestamp}.mp4"
        video_path = os.path.join("output", video_filename)
        
        logger.info(f"Creating video using Shotstack with S3 audio...")
        # Create video with S3 audio URL
        result_path = video_creator.create_video(
            request.audio_url,
            request.narration_lines,
            video_path
        )
        logger.info(f"Video created successfully: {result_path}")
        
        return {
            "success": True,
            "video_path": result_path,
            "audio_url": request.audio_url,
            "message": "Video created successfully with S3 audio"
        }
    except Exception as e:
        logger.error(f"Failed to create video with S3: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create video with S3: {str(e)}")

@app.post("/generate-images")
async def generate_images(request: GenerateImagesRequest):
    """Generate images using DALL-E and upload to S3"""
    logger.info(f"Image generation requested for {len(request.script_lines)} lines with style: {request.style}")
    
    if image_generator is None:
        logger.error("Image generator not available")
        raise HTTPException(status_code=503, detail="Image generator not available. Check OPENAI_API_KEY configuration.")
    
    try:
        logger.info(f"Generating images using DALL-E...")
        # Generate images for script lines
        image_urls = image_generator.generate_images_for_script(request.script_lines, request.style)
        
        # Filter out None values (failed generations)
        successful_urls = [url for url in image_urls if url is not None]
        
        if successful_urls:
            logger.info(f"Generated {len(successful_urls)} images successfully")
            return {
                "success": True,
                "image_urls": successful_urls,
                "total_requested": len(request.script_lines),
                "total_generated": len(successful_urls),
                "style": request.style,
                "message": f"Generated {len(successful_urls)} images successfully"
            }
        else:
            logger.error("Failed to generate any images")
            raise HTTPException(status_code=500, detail="Failed to generate any images")
            
    except Exception as e:
        logger.error(f"Failed to generate images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate images: {str(e)}")

@app.post("/create-video-with-images")
async def create_video_with_images(request: ImageVideoRequest):
    """Create video using Shotstack with S3 audio URL and S3 image URLs"""
    logger.info(f"Video creation with images requested for audio URL: {request.audio_url}")
    logger.info(f"Using {len(request.image_urls)} images")
    
    if video_creator is None:
        logger.error("Video creator not available")
        raise HTTPException(status_code=503, detail="Video creator not available.")
    
    try:
        # Create output video path with timestamp
        timestamp = int(time.time())
        video_filename = f"shorts_with_images_{timestamp}.mp4"
        video_path = os.path.join("output", video_filename)
        
        logger.info(f"Creating video using Shotstack with S3 audio and images...")
        # Create video with S3 audio URL and image URLs
        result_path = video_creator.create_video_with_images(
            request.audio_url,
            request.narration_lines,
            request.image_urls,
            video_path
        )
        logger.info(f"Video with images created successfully: {result_path}")
        
        return {
            "success": True,
            "video_path": result_path,
            "audio_url": request.audio_url,
            "image_urls": request.image_urls,
            "message": "Video created successfully with S3 audio and images"
        }
    except Exception as e:
        logger.error(f"Failed to create video with images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create video with images: {str(e)}")

@app.post("/create-video")
async def create_video(request: VideoRequest):
    """Create video using FFmpeg with audio, images, and subtitles"""
    logger.info(f"Video creation requested for audio: {request.audio_path}")
    
    if video_creator is None:
        logger.error("Video creator not available")
        raise HTTPException(status_code=503, detail="Video creator not available.")
    
    try:
        # Create output video path
        video_filename = f"shorts_{int(os.path.getmtime(request.audio_path))}.mp4"
        video_path = os.path.join("output", video_filename)
        
        logger.info(f"Creating video using Shotstack...")
        # Create video
        result_path = video_creator.create_video(
            request.audio_path,
            request.narration_lines,
            video_path
        )
        logger.info(f"Video created successfully: {result_path}")
        
        return {
            "success": True,
            "video_path": result_path,
            "message": "Video created successfully"
        }
    except Exception as e:
        logger.error(f"Failed to create video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create video: {str(e)}")

@app.post("/upload-to-youtube")
async def upload_to_youtube(request: UploadRequest):
    """Upload video to YouTube using YouTube Data API"""
    logger.info(f"YouTube upload requested for video: {request.video_path}")
    
    if youtube_uploader is None:
        logger.error("YouTube uploader not available")
        raise HTTPException(status_code=503, detail="YouTube uploader not available.")
    
    try:
        logger.info(f"Uploading video to YouTube...")
        # Upload video
        result = youtube_uploader.upload_video(
            request.video_path,
            request.title,
            request.description,
            request.tags
        )
        
        if result:
            logger.info(f"Video uploaded successfully: {result['video_id']}")
            return {
                "success": True,
                "video_id": result["video_id"],
                "title": result["title"],
                "url": result["url"],
                "message": "Video uploaded successfully (private)"
            }
        else:
            logger.error("Failed to upload video - no result returned")
            raise HTTPException(status_code=500, detail="Failed to upload video")
            
    except Exception as e:
        logger.error(f"Failed to upload to YouTube: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload to YouTube: {str(e)}")

@app.post("/make-video-public/{video_id}")
async def make_video_public(video_id: str):
    """Make a private video public"""
    try:
        success = youtube_uploader.update_video_privacy(video_id, "public")
        
        if success:
            return {
                "success": True,
                "message": f"Video {video_id} is now public"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update video privacy")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update video privacy: {str(e)}")

@app.get("/channel-info")
async def get_channel_info():
    """Get YouTube channel information"""
    try:
        channel_info = youtube_uploader.get_channel_info()
        
        if channel_info:
            return {
                "success": True,
                "channel": channel_info
            }
        else:
            raise HTTPException(status_code=404, detail="Channel information not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get channel info: {str(e)}")

@app.get("/available-voices")
async def get_available_voices():
    """Get available voices from ElevenLabs"""
    try:
        voices = voice_synthesizer.get_available_voices()
        return {
            "success": True,
            "voices": voices
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get voices: {str(e)}")

@app.get("/download/{file_type}/{filename}")
async def download_file(file_type: str, filename: str):
    """Download generated files"""
    try:
        if file_type == "audio":
            file_path = os.path.join("output", filename)
        elif file_type == "video":
            file_path = os.path.join("output", filename)
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        if os.path.exists(file_path):
            return FileResponse(file_path, filename=filename)
        else:
            raise HTTPException(status_code=404, detail="File not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")

@app.get("/logs")
async def get_logs(lines: int = 100):
    """Get recent application logs"""
    try:
        log_file = "app.log"
        if not os.path.exists(log_file):
            return {"logs": [], "message": "No log file found"}
        
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "logs": recent_lines,
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines)
        }
    except Exception as e:
        logger.error(f"Failed to read logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")

@app.post("/full-pipeline")
async def full_pipeline():
    """Run the complete pipeline: script → voice → video → upload"""
    logger.info("Full pipeline execution started")
    
    try:
        # Step 1: Generate script
        logger.info("Step 1: Generating script...")
        script = script_generator.generate_script()
        narration_texts = [line.text for line in script.narration]
        logger.info(f"Script generated: {script.title}")
        
        # Step 2: Generate voice
        logger.info("Step 2: Generating voice...")
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir="output") as tmp_file:
            audio_path = tmp_file.name
        
        voice_synthesizer.synthesize_voice(narration_texts, audio_path)
        logger.info(f"Voice generated: {audio_path}")
        
        # Step 3: Create video
        logger.info("Step 3: Creating video...")
        video_filename = f"shorts_{int(os.path.getmtime(audio_path))}.mp4"
        video_path = os.path.join("output", video_filename)
        
        narration_lines = [line.dict() for line in script.narration]
        video_creator.create_video(audio_path, narration_lines, video_path)
        logger.info(f"Video created: {video_path}")
        
        # Step 4: Upload to YouTube
        logger.info("Step 4: Uploading to YouTube...")
        description = f"🤔 {script.title}\n\n#shorts #funny #relatable\n\nWhy Would You - Daily relatable moments!"
        
        #upload_result = youtube_uploader.upload_video(
        ##    video_path,
        #    script.title,
       #     description,
       #     script.tags
       # )
        #logger.info(f"Video uploaded: {upload_result['video_id']}")
        
        logger.info("Full pipeline completed successfully")
        return {
            "success": True,
            "pipeline": {
                "script": {
                    "title": script.title,
                    "total_duration": script.total_duration,
                    "tags": script.tags
                },
                "audio_path": audio_path,
                "video_path": video_path,
                "upload": upload_result
            },
            "message": "Full pipeline completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

@app.post("/full-pipeline-s3")
async def full_pipeline_s3():
    """Run the complete S3-integrated pipeline: script → voice (S3) → video (S3) → upload"""
    logger.info("S3 Full pipeline execution started")
    
    if s3_uploader is None:
        logger.error("S3 uploader not available")
        raise HTTPException(status_code=503, detail="S3 uploader not available. Check AWS credentials configuration.")
    
    try:
        # Step 1: Generate script
        logger.info("Step 1: Generating script...")
        script = script_generator.generate_script()
        narration_texts = [line.text for line in script.narration]
        logger.info(f"Script generated: {script.title}")
        
        # Step 2: Generate voice and upload to S3
        logger.info("Step 2: Generating voice and uploading to S3...")
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir="output") as tmp_file:
            audio_path = tmp_file.name
        
        s3_audio_url = voice_synthesizer.synthesize_and_upload_to_s3(narration_texts, audio_path)
        if not s3_audio_url:
            raise Exception("Failed to generate voice or upload to S3")
        logger.info(f"Voice uploaded to S3: {s3_audio_url}")
        
        # Step 3: Create video with S3 audio
        logger.info("Step 3: Creating video with S3 audio...")
        timestamp = int(time.time())
        video_filename = f"shorts_{timestamp}.mp4"
        video_path = os.path.join("output", video_filename)
        
        # Convert script lines to narration format
        narration_lines = []
        current_time = 0.0
        for line in script.narration:
            duration = line.duration if hasattr(line, 'duration') else 3.0
            narration_lines.append({
                "text": line.text,
                "duration": duration,
                "start": current_time
            })
            current_time += duration
        
        result_video_path = video_creator.create_video(s3_audio_url, narration_lines, video_path)
        logger.info(f"Video created: {result_video_path}")
        
        return {
            "success": True,
            "script": script.dict(),
            "audio_url": s3_audio_url,
            "local_audio_path": audio_path,
            "video_path": result_video_path,
            "message": "S3 Full pipeline completed successfully"
        }
    except Exception as e:
        logger.error(f"S3 Full pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"S3 Full pipeline failed: {str(e)}")

@app.post("/full-pipeline-with-images")
async def full_pipeline_with_images():
    """Run the complete pipeline with images: script → voice (S3) → images (S3) → video (S3)"""
    logger.info("Full pipeline with images execution started")
    
    if s3_uploader is None:
        logger.error("S3 uploader not available")
        raise HTTPException(status_code=503, detail="S3 uploader not available. Check AWS credentials configuration.")
    
    if image_generator is None:
        logger.error("Image generator not available")
        raise HTTPException(status_code=503, detail="Image generator not available. Check OPENAI_API_KEY configuration.")
    
    try:
        # Step 1: Generate script
        logger.info("Step 1: Generating script...")
        script = script_generator.generate_script()
        narration_texts = [line.text for line in script.narration]
        logger.info(f"Script generated: {script.title}")
        
        # Step 2: Generate voice and upload to S3
        logger.info("Step 2: Generating voice and uploading to S3...")
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir="output") as tmp_file:
            audio_path = tmp_file.name
        
        s3_audio_url = voice_synthesizer.synthesize_and_upload_to_s3(narration_texts, audio_path)
        if not s3_audio_url:
            raise Exception("Failed to generate voice or upload to S3")
        logger.info(f"Voice uploaded to S3: {s3_audio_url}")
        
        # Step 3: Generate images and upload to S3
        logger.info("Step 3: Generating images and uploading to S3...")
        image_urls = image_generator.generate_images_for_script(narration_texts, "relatable")
        successful_image_urls = [url for url in image_urls if url is not None]
        logger.info(f"Generated {len(successful_image_urls)} images")
        
        # Step 4: Create video with S3 audio and images
        logger.info("Step 4: Creating video with S3 audio and images...")
        timestamp = int(time.time())
        video_filename = f"shorts_with_images_{timestamp}.mp4"
        video_path = os.path.join("output", video_filename)
        
        # Convert script lines to narration format
        narration_lines = []
        current_time = 0.0
        for line in script.narration:
            duration = line.duration if hasattr(line, 'duration') else 3.0
            narration_lines.append({
                "text": line.text,
                "duration": duration,
                "start": current_time
            })
            current_time += duration
        
        result_video_path = video_creator.create_video_with_images(s3_audio_url, narration_lines, successful_image_urls, video_path)
        logger.info(f"Video with images created: {result_video_path}")
        
        return {
            "success": True,
            "script": script.dict(),
            "audio_url": s3_audio_url,
            "local_audio_path": audio_path,
            "image_urls": successful_image_urls,
            "video_path": result_video_path,
            "message": "Full pipeline with images completed successfully"
        }
    except Exception as e:
        logger.error(f"Full pipeline with images failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Full pipeline with images failed: {str(e)}")

if __name__ == "__main__":
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting Why Would You YouTube Shorts Automation API")
    logger.info(f"Host: {host}, Port: {port}, Debug: {debug}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    ) 