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
from utils.ffmpeg_video_creator import FFmpegVideoCreator
from utils.youtube_uploader import YouTubeUploader
from utils.image_generator import ImageGenerator
from utils.stable_diffusion_generator import StableDiffusionGenerator

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
    ffmpeg_video_creator = FFmpegVideoCreator()
    logger.info("FFmpegVideoCreator initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize FFmpegVideoCreator: {e}")
    ffmpeg_video_creator = None

try:
    youtube_uploader = YouTubeUploader()
    logger.info("YouTubeUploader initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize YouTubeUploader: {e}")
    youtube_uploader = None

try:
    image_generator = ImageGenerator()
    logger.info("ImageGenerator initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize ImageGenerator: {e}")
    image_generator = None

try:
    stable_diffusion_generator = StableDiffusionGenerator()
    logger.info("StableDiffusionGenerator initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize StableDiffusionGenerator: {e}")
    stable_diffusion_generator = None

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

class ImageVideoRequest(BaseModel):
    audio_path: str
    narration_lines: List[dict]
    image_paths: List[str]

class GenerateImagesRequest(BaseModel):
    script_lines: List[str]
    style: str = "realistic"
    use_stable_diffusion: bool = False

class FullPipelineWithImagesRequest(BaseModel):
    use_ffmpeg: bool = False
    animation_type: str = "zoom_in"
    use_stable_diffusion: bool = False
    # Optional script data from generate-script response
    script_title: str = None
    script_narration: List[dict] = None
    script_tags: List[str] = None
    # If script data is provided, skip script generation
    use_existing_script: bool = False
    # Story type for script generation (if not using existing script)
    story_type: str = "motivation"
    # S3 upload options
    upload_to_s3: bool = True
    s3_folder: str = "youtube-shorts"

class ScriptRetryRequest(BaseModel):
    retry: bool = True
    max_retries: int = 3
    story_type: str = "motivation"

class GenerateScriptRequest(BaseModel):
    story_type: str = "motivation"

class S3UploadRequest(BaseModel):
    video_path: str
    s3_key: str = None
    folder: str = "youtube-shorts"

# Create necessary directories
logger.info("Creating necessary directories...")
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("output/images", exist_ok=True)
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
            "generate_script_with_retry": "/generate-script-with-retry",
            "available_story_types": "/available-story-types",
            "available_image_generators": "/available-image-generators",
            "generate_voice": "/generate-voice",
            "create_video": "/create-video",
            "upload_to_youtube": "/upload-to-youtube",
            "full_pipeline": "/full-pipeline",
            "generate_images": "/generate-images",
            "create_video_with_images": "/create-video-with-images",
            "full_pipeline_with_images": "/full-pipeline-with-images",
            "logs": "/logs",
            "docs": "/docs"
        },
        "features": {
            "script_generation": {
                "retry_support": True,
                "max_retries": 3,
                "existing_script_support": True,
                "story_types": [
                    "motivation", "horror", "real_life", "adventure", 
                    "comedy", "drama", "mystery", "romance", "sci_fi", "fantasy"
                ]
            },
            "image_generation": {
                "dalle_enabled": image_generator is not None,
                "stable_diffusion_enabled": stable_diffusion_generator is not None,
                "dalle_model": "DALL-E 3",
                "stable_diffusion_model": "SG161222/Realistic_Vision_V5.1_noVAE",
                "description": "AI-powered image generation using DALL-E 3 or Stable Diffusion, saved locally in output/images folder"
            },
            "video_creation": {
                "ffmpeg_support": ffmpeg_video_creator is not None,
                "shotstack_support": video_creator is not None,
                "animation_types": ["zoom_in", "zoom_out", "pan_left", "pan_right", "static"]
            }
        },
        "usage_examples": {
            "generate_script_with_story_type": {
                "method": "GET",
                "endpoint": "/generate-script?story_type=horror"
            },
            "generate_script_with_retry": {
                "method": "POST",
                "endpoint": "/generate-script-with-retry",
                "body": {"retry": True, "max_retries": 3, "story_type": "adventure"}
            },
            "full_pipeline_with_story_type": {
                "method": "POST", 
                "endpoint": "/full-pipeline-with-images",
                "body": {
                    "story_type": "comedy",
                    "use_ffmpeg": True,
                    "animation_type": "zoom_in",
                    "use_stable_diffusion": False
                }
            },
            "full_pipeline_with_existing_script": {
                "method": "POST", 
                "endpoint": "/full-pipeline-with-images",
                "body": {
                    "use_existing_script": True,
                    "script_title": "Your Script Title",
                    "script_narration": [{"text": "Your text", "duration": 3.0, "start": 0.0}],
                    "script_tags": ["tag1", "tag2"]
                }
            },
            "generate_images_with_stable_diffusion": {
                "method": "POST",
                "endpoint": "/generate-images",
                "body": {
                    "script_lines": ["A person looking confused", "Someone scratching their head"],
                    "style": "realistic",
                    "use_stable_diffusion": True
                }
            }
        }
    }

@app.get("/generate-script")
async def generate_script(story_type: str = "motivation"):
    """Generate a YouTube Shorts script using OpenAI GPT with specified story type"""
    logger.info(f"Script generation requested for story type: {story_type}")
    
    if script_generator is None:
        logger.error("Script generator not available")
        raise HTTPException(status_code=503, detail="Script generator not available. Check OPENAI_API_KEY configuration.")
    
    try:
        logger.info(f"Generating {story_type} script using OpenAI...")
        script = script_generator.generate_script(story_type)
        logger.info(f"Script generated successfully: {script.title}")
        return {
            "success": True,
            "script": {
                "title": script.title,
                "narration": [line.dict() for line in script.narration],
                "total_duration": script.total_duration,
                "tags": script.tags
            },
            "story_type": story_type,
            "message": f"{story_type.capitalize()} script generated successfully. Use /generate-script-with-retry to retry if needed."
        }
    except Exception as e:
        logger.error(f"Failed to generate script: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate script: {str(e)}")

@app.post("/generate-script-with-retry")
async def generate_script_with_retry(request: ScriptRetryRequest = ScriptRetryRequest()):
    """Generate a YouTube Shorts script with retry functionality and story type support"""
    logger.info(f"Script generation with retry requested for story type: {request.story_type}")
    
    if script_generator is None:
        logger.error("Script generator not available")
        raise HTTPException(status_code=503, detail="Script generator not available. Check OPENAI_API_KEY configuration.")
    
    scripts = []
    attempts = 0
    max_attempts = request.max_retries if request.retry else 1
    
    while attempts < max_attempts:
        attempts += 1
        logger.info(f"Script generation attempt {attempts}/{max_attempts} for {request.story_type}")
        
        try:
            script = script_generator.generate_script(request.story_type)
            logger.info(f"Script {attempts} generated successfully: {script.title}")
            
            script_data = {
                "title": script.title,
                "narration": [line.dict() for line in script.narration],
                "total_duration": script.total_duration,
                "tags": script.tags,
                "attempt": attempts,
                "story_type": request.story_type
            }
            
            scripts.append(script_data)
            
            # If retry is disabled or this is the last attempt, return the result
            if not request.retry or attempts >= max_attempts:
                return {
                    "success": True,
                    "scripts": scripts,
                    "total_attempts": attempts,
                    "final_script": script_data,
                    "story_type": request.story_type,
                    "message": f"{request.story_type.capitalize()} script generated successfully after {attempts} attempt(s)"
                }
            
            # If retry is enabled and we have more attempts, continue
            logger.info(f"Script generated, but retry is enabled. Continuing to next attempt...")
            
        except Exception as e:
            logger.error(f"Failed to generate script on attempt {attempts}: {str(e)}")
            
            if attempts >= max_attempts:
                raise HTTPException(status_code=500, detail=f"Failed to generate script after {attempts} attempts: {str(e)}")
            
            logger.info(f"Retrying script generation...")
    
    # This should not be reached, but just in case
    raise HTTPException(status_code=500, detail="Unexpected error in script generation with retry")

@app.post("/generate-voice")
async def generate_voice(request: VoiceRequest):
    """Generate voiceover using ElevenLabs API"""
    logger.info(f"Voice generation requested for {len(request.narration)} lines")
    
    if voice_synthesizer is None:
        logger.error("Voice synthesizer not available")
        raise HTTPException(status_code=503, detail="Voice synthesizer not available. Check ELEVENLABS_API_KEY configuration.")
    
    try:
        # Create output file path with timestamp
        timestamp = int(time.time())
        audio_filename = f"voice_{timestamp}.mp3"
        audio_path = os.path.join("output", audio_filename)
        
        logger.info(f"Generating voice...")
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

@app.post("/generate-images")
async def generate_images(request: GenerateImagesRequest):
    """Generate images using DALL-E or Stable Diffusion and save locally"""
    logger.info(f"Image generation requested for {len(request.script_lines)} lines with style: {request.style}")
    logger.info(f"Using {'Stable Diffusion' if request.use_stable_diffusion else 'DALL-E'} for image generation")
    
    # Choose image generator based on switch
    if request.use_stable_diffusion:
        if stable_diffusion_generator is None:
            logger.error("Stable Diffusion generator not available")
            raise HTTPException(status_code=503, detail="Stable Diffusion generator not available. Check model configuration.")
        generator = stable_diffusion_generator
        generator_name = "Stable Diffusion"
    else:
        if image_generator is None:
            logger.error("DALL-E image generator not available")
            raise HTTPException(status_code=503, detail="DALL-E image generator not available. Check OPENAI_API_KEY configuration.")
        generator = image_generator
        generator_name = "DALL-E"
    
    try:
        logger.info(f"Generating images using {generator_name}...")
        # Generate images for script lines
        image_paths = generator.generate_images_for_script(request.script_lines, request.style)
        
        # Filter out None values (failed generations)
        successful_paths = [path for path in image_paths if path is not None]
        
        if successful_paths:
            logger.info(f"Generated {len(successful_paths)} images successfully using {generator_name}")
            return {
                "success": True,
                "image_paths": successful_paths,
                "total_requested": len(request.script_lines),
                "total_generated": len(successful_paths),
                "style": request.style,
                "generator_used": generator_name,
                "message": f"Generated {len(successful_paths)} images successfully using {generator_name}"
            }
        else:
            logger.error(f"Failed to generate any images using {generator_name}")
            raise HTTPException(status_code=500, detail=f"Failed to generate any images using {generator_name}")
            
    except Exception as e:
        logger.error(f"Failed to generate images using {generator_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate images using {generator_name}: {str(e)}")

@app.post("/create-video-with-images")
async def create_video_with_images(request: ImageVideoRequest):
    """Create video using Shotstack with local audio path and local image paths"""
    logger.info(f"Video creation with images requested for audio path: {request.audio_path}")
    logger.info(f"Using {len(request.image_paths)} images")
    
    if video_creator is None:
        logger.error("Video creator not available")
        raise HTTPException(status_code=503, detail="Video creator not available.")
    
    try:
        # Create output video path with timestamp
        timestamp = int(time.time())
        video_filename = f"shorts_with_images_{timestamp}.mp4"
        video_path = os.path.join("output", video_filename)
        
        logger.info(f"Creating video using Shotstack with local audio and images...")
        # Create video with local audio path and image paths
        result_path = video_creator.create_video_with_images(
            request.audio_path,
            request.narration_lines,
            request.image_paths,
            video_path
        )
        logger.info(f"Video created successfully: {result_path}")
        
        return {
            "success": True,
            "video_path": result_path,
            "audio_path": request.audio_path,
            "image_paths": request.image_paths,
            "message": "Video created successfully with local audio and images"
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

@app.get("/available-story-types")
async def get_available_story_types():
    """Get available story types for script generation"""
    try:
        story_types = script_generator.get_available_story_types()
        return {
            "success": True,
            "story_types": story_types
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get story types: {str(e)}")

@app.get("/available-image-generators")
async def get_available_image_generators():
    """Get available image generators and their status"""
    try:
        return {
            "success": True,
            "generators": {
                "dalle": {
                    "name": "DALL-E 3",
                    "enabled": image_generator is not None,
                    "description": "OpenAI's DALL-E 3 model for high-quality image generation",
                    "requires_api_key": "OPENAI_API_KEY"
                },
                "stable_diffusion": {
                    "name": "Stable Diffusion",
                    "enabled": stable_diffusion_generator is not None,
                    "description": "Open-source Stable Diffusion model for local image generation",
                    "model_id": "SG161222/Realistic_Vision_V5.1_noVAE",
                    "requires": "PyTorch, diffusers library"
                }
            },
            "default": "dalle",
            "usage": {
                "generate_images": "Set use_stable_diffusion=true to use Stable Diffusion",
                "full_pipeline": "Set use_stable_diffusion=true to use Stable Diffusion"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get image generators: {str(e)}")

@app.post("/upload-to-s3")
async def upload_to_s3(request: S3UploadRequest):
    """Upload a video to S3 bucket"""
    try:
        if not ffmpeg_video_creator:
            raise HTTPException(status_code=500, detail="FFmpeg video creator not available")
        
        if not os.path.exists(request.video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_path}")
        
        logger.info(f"Uploading video to S3: {request.video_path}")
        
        # Upload to S3
        s3_url = ffmpeg_video_creator.upload_to_s3(
            video_path=request.video_path,
            s3_key=request.s3_key,
            folder=request.folder
        )
        
        if s3_url:
            return {
                "success": True,
                "message": "Video uploaded to S3 successfully",
                "s3_url": s3_url,
                "local_path": request.video_path,
                "s3_key": request.s3_key or f"{request.folder}/{os.path.basename(request.video_path)}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to upload video to S3")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading to S3: {str(e)}")

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

@app.post("/full-pipeline-with-images")
async def full_pipeline_with_images(request: FullPipelineWithImagesRequest = FullPipelineWithImagesRequest()):
    """Run the complete pipeline with images: script → voice → images → video → upload"""
    logger.info("Full pipeline with images execution started")
    
    try:
        # Step 1: Generate script or use existing script
        if request.use_existing_script and request.script_narration:
            logger.info("Step 1: Using existing script from request...")
            # Create a script object from the provided data
            from utils.script_generator import GeneratedScript, ScriptLine
            
            # Convert narration data to ScriptLine objects
            narration_lines = []
            for line_data in request.script_narration:
                narration_lines.append(ScriptLine(
                    text=line_data.get('text', ''),
                    duration=line_data.get('duration', 3.0),
                    start=line_data.get('start', 0.0)
                ))
            
            # Create a GeneratedScript object
            script = GeneratedScript(
                title=request.script_title or "Custom Script",
                narration=narration_lines,
                tags=request.script_tags or []
            )
            logger.info(f"Using existing script: {script.title}")
        else:
            logger.info("Step 1: Generating new script...")
            script = script_generator.generate_script(request.story_type)
            logger.info(f"Script generated: {script.title}")
        
        narration_texts = [line.text for line in script.narration]
        visual_texts = [line.visual_suggestion for line in script.narration]
        
        # Step 2: Generate voice
        logger.info("Step 2: Generating voice...")
        timestamp = int(time.time())
        audio_filename = f"voice_{timestamp}.mp3"
        audio_path = os.path.join("output", audio_filename)
        
        voice_synthesizer.synthesize_voice(narration_texts, audio_path)
        logger.info(f"Voice generated: {audio_path}")
        
        # Step 3: Generate images
        logger.info("Step 3: Generating images...")
        logger.info(f"Using {'Stable Diffusion' if request.use_stable_diffusion else 'DALL-E'} for image generation")
        
        # Choose image generator based on switch
        if request.use_stable_diffusion:
            if stable_diffusion_generator is None:
                raise HTTPException(status_code=503, detail="Stable Diffusion generator not available. Check model configuration.")
            generator = stable_diffusion_generator
            generator_name = "Stable Diffusion"
        else:
            if image_generator is None:
                raise HTTPException(status_code=503, detail="DALL-E image generator not available. Check OPENAI_API_KEY configuration.")
            generator = image_generator
            generator_name = "DALL-E"
        
        image_paths = generator.generate_images_for_script(visual_texts, "relatable")
        successful_image_paths = [path for path in image_paths if path is not None]
        logger.info(f"Generated {len(successful_image_paths)} images using {generator_name}")
        
        # Step 4: Create video with audio and images
        logger.info("Step 4: Creating video with audio and images...")
        video_filename = f"shorts_with_images_{timestamp}.mp4"
        video_path = os.path.join("output", video_filename)
        logger.info(f"Video path: {video_path} and audio path: {audio_path} filename: {video_filename}")
        # Convert script lines to narration format for FFmpeg
        ffmpeg_narration_lines = []
        for i, line in enumerate(script.narration):
            duration = line.duration if hasattr(line, 'duration') else 3.0
            ffmpeg_narration_lines.append({
                "text": line.text,
                "duration": duration,
                "visual_suggestion": successful_image_paths[i] if i < len(successful_image_paths) else successful_image_paths[0]
            })
            logger.info(f"FFmpeg narration line: {ffmpeg_narration_lines}")
        
        # Convert script lines to narration format for Shotstack
        shotstack_narration_lines = []
        current_time = 0.0
        for line in script.narration:
            duration = line.duration if hasattr(line, 'duration') else 3.0
            shotstack_narration_lines.append({
                "text": line.text,
                "duration": duration,
                "start": current_time
            })
            current_time += duration
        
        # Choose video creator based on switch
        if request.use_ffmpeg:
            if ffmpeg_video_creator is None:
                raise HTTPException(status_code=503, detail="FFmpegVideoCreator not available")
            
            # Choose between different video creation methods
            if request.use_stable_diffusion and ffmpeg_video_creator.sd_generator:
                logger.info(f"Using FFmpegVideoCreator with Stable Video Diffusion for motion videos")
                result_video_path = ffmpeg_video_creator.create_video_with_motion_images(
                    audio_path=audio_path,
                    narration_lines=ffmpeg_narration_lines,
                    image_paths=successful_image_paths,
                    output_path=video_path,
                    motion_strength=1.0,  # Enhanced motion strength for more visible effects
                    num_frames_per_segment=12,  # Reduced for faster generation
                    video_fps=8,
                    fast_mode=True,  # Enable fast mode
                    timeout_seconds=60,  # 60 second timeout per segment
                    motion_type="dynamic"  # Use dynamic motion for maximum effect
                )
            elif request.animation_type in ["zoom_pan", "rotate", "scale"]:
                # Use FFmpeg motion effects for faster generation
                logger.info(f"Using FFmpegVideoCreator with FFmpeg motion effects: {request.animation_type}")
                result_video_path = ffmpeg_video_creator.create_video_with_ffmpeg_motion(
                    audio_path=audio_path,
                    narration_lines=ffmpeg_narration_lines,
                    image_paths=successful_image_paths,
                    output_path=video_path,
                    motion_type=request.animation_type
                )
            else:
                # Use static images with FFmpeg animations
                logger.info(f"Using FFmpegVideoCreator for video creation with {request.animation_type} animation")
                result_video_path = ffmpeg_video_creator.create_video_with_images(
                    audio_path=audio_path,
                    narration_lines=ffmpeg_narration_lines,
                    image_paths=successful_image_paths,
                    output_path=video_path,
                    animation_type=request.animation_type
                )
        else:
            if video_creator is None:
                raise HTTPException(status_code=503, detail="ShotstackVideoCreator not available")
            logger.info("Using ShotstackVideoCreator for video creation")
            result_video_path = video_creator.create_video_with_images(audio_path, shotstack_narration_lines, successful_image_paths, video_path)
        
        logger.info(f"Video with images created: {result_video_path}")
        
        # Step 5: Upload to S3 (optional)
        s3_url = None
        if request.upload_to_s3 and ffmpeg_video_creator:
            try:
                logger.info("Step 5: Uploading video to S3...")
                s3_url = ffmpeg_video_creator.upload_to_s3(
                    video_path=result_video_path,
                    folder=request.s3_folder
                )
                if s3_url:
                    logger.info(f"Video uploaded to S3: {s3_url}")
                else:
                    logger.warning("Failed to upload video to S3")
            except Exception as e:
                logger.error(f"Error uploading to S3: {e}")
        
        return {
            "success": True,
            "script": script.dict(),
            "audio_path": audio_path,
            "image_paths": successful_image_paths,
            "video_path": result_video_path,
            "video_creator": "FFmpegVideoCreator" if request.use_ffmpeg else "ShotstackVideoCreator",
            "animation_type": request.animation_type if request.use_ffmpeg else "N/A",
            "image_generator": generator_name,
            "script_source": "existing" if request.use_existing_script else "generated",
            "s3_url": s3_url,
            "uploaded_to_s3": s3_url is not None,
            "message": "Full pipeline with images completed successfully"
        }
    except Exception as e:
        logger.error(f"Full pipeline with images failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Full pipeline with images failed: {str(e)}")

import atexit
import signal

def cleanup_resources():
    """Clean up resources on application shutdown"""
    logger.info("Cleaning up resources...")
    if stable_diffusion_generator is not None:
        try:
            stable_diffusion_generator.cleanup()
            logger.info("StableDiffusionGenerator cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up StableDiffusionGenerator: {e}")

# Register cleanup function
atexit.register(cleanup_resources)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    cleanup_resources()
    exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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