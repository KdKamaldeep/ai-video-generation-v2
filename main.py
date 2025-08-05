import os
import tempfile
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# Import utility modules
from utils.script_generator import ScriptGenerator, GeneratedScript, ScriptLine
from utils.voice_synthesizer import VoiceSynthesizer
from utils.video_creator import VideoCreator
from utils.youtube_uploader import YouTubeUploader

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
    video_creator = VideoCreator()
    logger.info("VideoCreator initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize VideoCreator: {e}")
    video_creator = None

try:
    youtube_uploader = YouTubeUploader()
    logger.info("YouTubeUploader initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize YouTubeUploader: {e}")
    youtube_uploader = None

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
            "upload_to_youtube": "/upload-to-youtube"
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
        
        logger.info(f"Creating video using FFmpeg...")
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
        
        upload_result = youtube_uploader.upload_video(
            video_path,
            script.title,
            description,
            script.tags
        )
        logger.info(f"Video uploaded: {upload_result['video_id']}")
        
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