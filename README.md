# Why Would You - YouTube Shorts Automation

A complete FastAPI backend for automating YouTube Shorts creation for the "Why Would You" channel. This application generates scripts, creates voiceovers, produces videos, and uploads them to YouTube automatically.

## 🚀 Features

- **Script Generation**: Uses OpenAI GPT to create engaging <60s YouTube Shorts scripts
- **Voice Synthesis**: Converts narration to speech using ElevenLabs API
- **Video Creation**: Generates vertical videos using Shotstack API, including subtitles and background images
- **YouTube Upload**: Automatically uploads videos to YouTube with proper metadata
- **Full Pipeline**: Complete automation from script to published video
- **Docker Support**: Full containerization with Docker and Docker Compose

## 📋 Prerequisites

- Python 3.11+
- API Keys for:
  - OpenAI GPT
  - ElevenLabs
  - Shotstack
  - YouTube Data API

## 🛠️ Installation

### Option 1: Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd WhyWouldYou
   ```

2. **Set up environment variables**:
   ```bash
   cp env.template .env
   # Edit .env with your API keys
   ```

3. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

### Option 2: Local Development

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp env.template .env
   # Edit .env with your API keys
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `env.template`:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# ElevenLabs API Configuration
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id_here

# Shotstack API Configuration
SHOTSTACK_API_KEY=your_shotstack_api_key_here

# YouTube API Configuration
YOUTUBE_CLIENT_ID=your_youtube_client_id_here
YOUTUBE_CLIENT_SECRET=your_youtube_client_secret_here
YOUTUBE_REDIRECT_URI=http://localhost:8000/auth/callback

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=True

# File Storage
UPLOAD_DIR=./uploads
OUTPUT_DIR=./output
```

### Getting API Keys

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **ElevenLabs API Key**: Get from [ElevenLabs](https://elevenlabs.io/)
3. **Shotstack API Key**: Get from [Shotstack](https://shotstack.io/)
4. **YouTube API**: 
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a project and enable YouTube Data API v3
   - Create OAuth 2.0 credentials
   - Download the client configuration

## 📚 API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

### Core Endpoints

#### 1. Generate Script
```http
GET /generate-script
```
Generates a YouTube Shorts script using OpenAI GPT.

**Response**:
```json
{
  "success": true,
  "script": {
    "title": "Why Would You Leave Your Phone on Silent?",
    "narration": [
      {
        "text": "Why would you leave your phone on silent?",
        "duration": 3.0,
        "visual_suggestion": "person frantically searching for phone"
      }
    ],
    "total_duration": 10.5,
    "tags": ["funny", "relatable", "phone", "shorts"]
  }
}
```

#### 2. Generate Voice
```http
POST /generate-voice
Content-Type: application/json

{
  "narration": ["Why would you leave your phone on silent?"]
}
```

#### 3. Create Video
```http
POST /create-video
Content-Type: application/json

{
  "audio_path": "/path/to/audio.mp3",
  "narration_lines": [
    {
      "text": "Why would you leave your phone on silent?",
      "duration": 3.0,
      "visual_suggestion": "person frantically searching for phone"
    }
  ]
}
```

#### 4. Upload to YouTube
```http
POST /upload-to-youtube
Content-Type: application/json

{
  "video_path": "/path/to/video.mp4",
  "title": "Why Would You Leave Your Phone on Silent?",
  "description": "🤔 Relatable moments...",
  "tags": ["funny", "relatable", "shorts"]
}
```

#### 5. Full Pipeline
```http
POST /full-pipeline
```
Runs the complete automation pipeline: script → voice → video → upload.

### Additional Endpoints

- `GET /channel-info` - Get YouTube channel information
- `GET /available-voices` - List available ElevenLabs voices
- `POST /make-video-public/{video_id}` - Make a private video public
- `GET /download/{file_type}/{filename}` - Download generated files

## 🎬 Usage Examples

### Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Generate a script
response = requests.get(f"{BASE_URL}/generate-script")
script_data = response.json()["script"]

# Generate voice
narration = [line["text"] for line in script_data["narration"]]
response = requests.post(f"{BASE_URL}/generate-voice", 
                        json={"narration": narration})
audio_path = response.json()["audio_path"]

# Create video
response = requests.post(f"{BASE_URL}/create-video",
                        json={
                            "audio_path": audio_path,
                            "narration_lines": script_data["narration"]
                        })
video_path = response.json()["video_path"]

# Upload to YouTube
response = requests.post(f"{BASE_URL}/upload-to-youtube",
                        json={
                            "video_path": video_path,
                            "title": script_data["title"],
                            "description": f"🤔 {script_data['title']}",
                            "tags": script_data["tags"]
                        })
print(f"Video uploaded: {response.json()['url']}")
```

### cURL Examples

```bash
# Generate script
curl -X GET "http://localhost:8000/generate-script"

# Run full pipeline
curl -X POST "http://localhost:8000/full-pipeline"

# Get channel info
curl -X GET "http://localhost:8000/channel-info"
```

## 🐳 Docker Commands

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild without cache
docker-compose build --no-cache
```

## 📁 Project Structure

```
WhyWouldYou/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── .dockerignore          # Docker ignore file
├── env.template           # Environment template
├── README.md              # This file
└── utils/                 # Utility modules
    ├── __init__.py
    ├── script_generator.py    # OpenAI GPT integration
    ├── voice_synthesizer.py   # ElevenLabs integration
    ├── video_creator.py       # FFmpeg video creation
    └── youtube_uploader.py    # YouTube API integration
```

## 🔒 Security Considerations

- Store API keys securely in environment variables
- Videos are uploaded as private by default
- OAuth2 tokens are stored locally in `token.pickle`
- Use HTTPS in production environments

## 🚨 Troubleshooting

### Common Issues

1. **FFmpeg not found**: Ensure FFmpeg is installed or use Docker
2. **API key errors**: Verify all API keys are correctly set in `.env`
3. **YouTube authentication**: First run will require browser authentication
4. **Memory issues**: Large video files may require more RAM

### Logs

```bash
# View application logs
docker-compose logs youtube-shorts-automation

# View specific service logs
docker-compose logs -f youtube-shorts-automation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and personal use. Ensure compliance with:
- YouTube's Terms of Service
- OpenAI's Usage Policies
- ElevenLabs' Terms of Service
- Local laws and regulations

Always review generated content before publishing. 