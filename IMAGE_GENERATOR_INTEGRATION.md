# Image Generator Integration

This document explains the integration of both DALL-E 3 and Stable Diffusion image generators in the Why Would You YouTube Shorts Automation API.

## Overview

The API now supports two image generation options:
- **DALL-E 3**: OpenAI's high-quality image generation model (requires API key)
- **Stable Diffusion**: Open-source local image generation model (requires PyTorch)

## Configuration

### DALL-E 3 Setup
1. Set your OpenAI API key in `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Stable Diffusion Setup
1. Install required dependencies:
   ```bash
   pip install torch diffusers transformers accelerate
   ```

2. The model will be automatically downloaded on first use (SG161222/Realistic_Vision_V5.1_noVAE)
   - This is a high-quality realistic image generation model
   - Optimized for photorealistic human faces and scenes
   - No VAE included (uses default VAE for better compatibility)

## API Usage

### Check Available Generators
```bash
GET /available-image-generators
```

Response:
```json
{
  "success": true,
  "generators": {
    "dalle": {
      "name": "DALL-E 3",
      "enabled": true,
      "description": "OpenAI's DALL-E 3 model for high-quality image generation",
      "requires_api_key": "OPENAI_API_KEY"
    },
    "stable_diffusion": {
      "name": "Stable Diffusion",
      "enabled": true,
      "description": "Open-source Stable Diffusion model for local image generation",
             "model_id": "SG161222/Realistic_Vision_V5.1_noVAE",
      "requires": "PyTorch, diffusers library"
    }
  },
  "default": "dalle"
}
```

### Generate Images with DALL-E (Default)
```bash
POST /generate-images
Content-Type: application/json

{
  "script_lines": [
    "A person looking confused and scratching their head",
    "Someone with a puzzled expression"
  ],
  "style": "realistic",
  "use_stable_diffusion": false
}
```

### Generate Images with Stable Diffusion
```bash
POST /generate-images
Content-Type: application/json

{
  "script_lines": [
    "A person looking confused and scratching their head",
    "Someone with a puzzled expression"
  ],
  "style": "realistic",
  "use_stable_diffusion": true
}
```

### Full Pipeline with DALL-E
```bash
POST /full-pipeline-with-images
Content-Type: application/json

{
  "story_type": "comedy",
  "use_ffmpeg": false,
  "use_stable_diffusion": false
}
```

### Full Pipeline with Stable Diffusion
```bash
POST /full-pipeline-with-images
Content-Type: application/json

{
  "story_type": "comedy",
  "use_ffmpeg": false,
  "use_stable_diffusion": true
}
```

## Request Parameters

### GenerateImagesRequest
- `script_lines`: List of text descriptions for image generation
- `style`: Image style (realistic, cinematic, artistic, etc.)
- `use_stable_diffusion`: Boolean to switch between generators (default: false)

### FullPipelineWithImagesRequest
- `use_stable_diffusion`: Boolean to switch between generators (default: false)
- `use_ffmpeg`: Boolean to use FFmpeg instead of Shotstack (default: false)
- `animation_type`: Animation type for FFmpeg (zoom_in, zoom_out, pan_left, pan_right, static)
- `story_type`: Story type for script generation
- `use_existing_script`: Boolean to use existing script data
- `script_title`, `script_narration`, `script_tags`: Existing script data

## Response Format

### Generate Images Response
```json
{
  "success": true,
  "image_paths": ["output/images/20231201_123456_abc123_image1.png"],
  "total_requested": 2,
  "total_generated": 2,
  "style": "realistic",
  "generator_used": "DALL-E",
  "message": "Generated 2 images successfully using DALL-E"
}
```

### Full Pipeline Response
```json
{
  "success": true,
  "script": { ... },
  "audio_path": "output/voice_1701234567.mp3",
  "image_paths": ["output/images/..."],
  "video_path": "output/shorts_with_images_1701234567.mp4",
  "video_creator": "ShotstackVideoCreator",
  "animation_type": "N/A",
  "image_generator": "Stable Diffusion",
  "script_source": "generated",
  "message": "Full pipeline with images completed successfully"
}
```

## Features

### DALL-E 3 Features
- High-quality image generation
- Multiple style options
- Fast generation
- Requires internet connection and API key

### Stable Diffusion Features
- Local image generation (no API costs)
- Character consistency support
- Multiple style options
- Customizable parameters (steps, guidance scale, etc.)
- Requires GPU for optimal performance
- Uses Realistic Vision V5.1 model for high-quality realistic images

### Character Consistency (Stable Diffusion Only)
The Stable Diffusion generator supports character consistency across multiple images:

```python
# Set character consistency
stable_diffusion_generator.set_character_consistency(
    "A young woman with brown hair and glasses",
    seed=12345
)

# Generate consistent character images
image_paths = stable_diffusion_generator.generate_consistent_character_frames(
    script_lines,
    "A young woman with brown hair and glasses",
    style="realistic",
    num_variations=3
)
```

## Error Handling

The API provides clear error messages for different scenarios:

- **Generator not available**: Check configuration and dependencies
- **API key missing**: Set required environment variables
- **Model loading failed**: Check internet connection and disk space
- **Generation failed**: Check input parameters and try again

## Testing

Run the test script to verify integration:

```bash
python test_image_generators.py
```

This will test:
- Available generators endpoint
- DALL-E image generation
- Stable Diffusion image generation
- Full pipeline with both generators

## Performance Considerations

### DALL-E 3
- Fast generation (10-30 seconds per image)
- Consistent quality
- API rate limits apply

### Stable Diffusion
- Slower generation (30-120 seconds per image)
- Quality depends on hardware
- No API costs or rate limits
- GPU acceleration recommended

## Troubleshooting

### DALL-E Issues
1. Check `OPENAI_API_KEY` environment variable
2. Verify internet connection
3. Check API quota and billing

### Stable Diffusion Issues
1. Install PyTorch and diffusers: `pip install torch diffusers transformers accelerate`
2. Ensure sufficient disk space for model download (~4GB for Realistic Vision V5.1)
3. Use GPU if available for better performance
4. Check CUDA installation for GPU support
5. Realistic Vision model is optimized for realistic human faces and scenes

### Memory Issues
- Reduce batch size for Stable Diffusion
- Use attention slicing (enabled by default)
- Close other applications to free memory

## Examples

### Basic Image Generation
```python
import requests

# Generate with DALL-E
response = requests.post("http://localhost:8000/generate-images", json={
    "script_lines": ["A confused person"],
    "style": "realistic",
    "use_stable_diffusion": False
})

# Generate with Stable Diffusion
response = requests.post("http://localhost:8000/generate-images", json={
    "script_lines": ["A confused person"],
    "style": "realistic",
    "use_stable_diffusion": True
})
```

### Full Pipeline Example
```python
# Complete pipeline with Stable Diffusion
response = requests.post("http://localhost:8000/full-pipeline-with-images", json={
    "story_type": "comedy",
    "use_stable_diffusion": True,
    "use_ffmpeg": True,
    "animation_type": "zoom_in"
})
```

## Migration Guide

### From Previous Version
If you were using the previous version with only DALL-E:

1. **No breaking changes**: DALL-E remains the default
2. **New parameter**: Add `use_stable_diffusion: false` to explicitly use DALL-E
3. **New endpoint**: Use `/available-image-generators` to check generator status

### Example Migration
```python
# Old way (still works)
response = requests.post("/generate-images", json={
    "script_lines": ["A person"],
    "style": "realistic"
})

# New way (explicit)
response = requests.post("/generate-images", json={
    "script_lines": ["A person"],
    "style": "realistic",
    "use_stable_diffusion": False  # Explicit DALL-E usage
})
``` 