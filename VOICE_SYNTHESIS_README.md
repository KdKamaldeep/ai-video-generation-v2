# Voice Synthesis System

This system provides unified voice synthesis capabilities using multiple providers, specifically designed for kids content and YouTube Shorts generation.

## 🎤 Available Providers

### 1. **Coqui TTS Bark** (Recommended)
- **Type**: Local open-source
- **Quality**: High
- **Cost**: Free
- **Features**: Voice cloning, offline operation
- **Best for**: Development, privacy, cost-effective production

### 2. **ElevenLabs API**
- **Type**: Cloud-based API
- **Quality**: Very High
- **Cost**: Paid API
- **Features**: Advanced voice cloning, high-quality output
- **Best for**: Production, high-quality content

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For Coqui TTS (recommended)
pip install TTS>=0.22.0

# For ElevenLabs (optional)
# Set ELEVENLABS_API_KEY in your environment
```

### Basic Usage

```python
from utils.unified_voice_synthesizer import UnifiedVoiceSynthesizer

# Initialize with Coqui TTS (default)
synthesizer = UnifiedVoiceSynthesizer(provider="coqui")

# Generate voice
text_lines = ["Hello, this is a test of voice synthesis."]
audio_file = synthesizer.synthesize_voice(text_lines, "output.wav")

# Cleanup
synthesizer.cleanup()
```

## 🔧 Configuration

### Environment Variables

```bash
# Voice provider selection
VOICE_PROVIDER=coqui  # or "elevenlabs"

# ElevenLabs configuration (if using ElevenLabs)
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_VOICE_ID=your_voice_id_here
```

### Provider-Specific Configuration

#### Coqui TTS Configuration
```python
coqui_config = {
    "gpu": True,                    # Use GPU (recommended)
    "speaker": "random",            # Speaker selection
    "voice_dir": "bark_voices/",    # Voice storage directory
    "text_temp": 0.7,              # Text generation temperature
    "waveform_temp": 0.7           # Audio generation temperature
}

synthesizer = UnifiedVoiceSynthesizer(
    provider="coqui",
    config=coqui_config
)
```

#### ElevenLabs Configuration
```python
# ElevenLabs uses environment variables for configuration
# No additional config needed in code
synthesizer = UnifiedVoiceSynthesizer(provider="elevenlabs")
```

## 🎭 Voice Cloning (Coqui TTS)

### Clone a Voice
```python
# Clone voice from audio file
success = synthesizer.clone_voice(
    audio_file_path="sample_voice.wav",
    speaker_name="my_voice",
    test_text="Hello, this is a test of the cloned voice."
)

if success:
    # Use cloned voice
    audio_file = synthesizer.synthesize_voice(
        ["Hello from cloned voice!"],
        "cloned_output.wav",
        speaker="my_voice"
    )
```

### Voice Cloning Requirements
- **Audio file**: WAV or MP3 format
- **Duration**: 10-30 seconds recommended
- **Quality**: Clear speech, minimal background noise
- **Language**: English recommended for best results

## 🔄 Provider Switching

### Switch Between Providers
```python
# Start with Coqui TTS
synthesizer = UnifiedVoiceSynthesizer(provider="coqui")

# Switch to ElevenLabs
synthesizer.switch_provider("elevenlabs")

# Switch back to Coqui TTS
synthesizer.switch_provider("coqui", {
    "gpu": True,
    "speaker": "random"
})
```

### Get Provider Information
```python
info = synthesizer.get_provider_info()
print(f"Current provider: {info['provider']}")
print(f"Type: {info['type']}")
print(f"Offline: {info['offline']}")
print(f"Voice cloning: {info['voice_cloning']}")
```

## 👶 Kids Content Optimization

### Kids Voice Configuration
```python
# Optimized for kids content
kids_config = {
    "gpu": True,
    "speaker": "random",
    "text_temp": 0.8,      # More expressive
    "waveform_temp": 0.7   # Balanced quality
}

synthesizer = UnifiedVoiceSynthesizer(
    provider="coqui",
    config=kids_config
)

# Kids-friendly text
kids_lines = [
    "Hello little friends!",
    "Let's learn something fun today!",
    "Are you ready for an adventure?"
]

audio_file = synthesizer.synthesize_voice(kids_lines, "kids_voice.wav")
```

## 📁 File Structure

```
utils/
├── unified_voice_synthesizer.py    # Main unified interface
├── coqui_voice_synthesizer.py      # Coqui TTS implementation
├── voice_synthesizer.py            # ElevenLabs implementation
└── ...

bark_voices/                        # Coqui TTS voice storage
├── speaker1/
│   ├── speaker.wav
│   └── speaker.npz
└── speaker2/
    ├── speaker.wav
    └── speaker.npz
```

## 🧪 Testing

### Run Examples
```bash
# Test voice provider switching
python voice_provider_example.py

# Test Coqui TTS specifically
python utils/coqui_voice_synthesizer.py

# Test unified interface
python utils/unified_voice_synthesizer.py
```

### Test Kids Cartoon Generation
```bash
# Generate kids cartoon with voice
python kids_cartoon_generator.py

# Test kids voice specifically
python kids_cartoon_example.py
```

## 📊 Provider Comparison

| Feature | Coqui TTS | ElevenLabs |
|---------|-----------|------------|
| **Type** | Local open-source | Cloud API |
| **Quality** | High | Very High |
| **Speed** | Medium (GPU recommended) | Fast |
| **Cost** | Free | Paid API |
| **Voice Cloning** | Basic | Advanced |
| **Offline** | ✅ Yes | ❌ No |
| **Setup** | Model download | API key |
| **Best For** | Development, privacy | Production |

## 🔧 Troubleshooting

### Common Issues

#### Coqui TTS Issues
```bash
# Install TTS with GPU support
pip install TTS[all]

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Clear model cache
rm -rf ~/.local/share/tts/
```

#### ElevenLabs Issues
```bash
# Check API key
echo $ELEVENLABS_API_KEY

# Test API connection
curl -H "xi-api-key: $ELEVENLABS_API_KEY" \
     https://api.elevenlabs.io/v1/voices
```

### Performance Optimization

#### For Coqui TTS
- Use GPU for faster generation
- Adjust temperature parameters for quality/speed balance
- Use appropriate speaker selection

#### For ElevenLabs
- Use appropriate voice ID for your content
- Batch requests when possible
- Monitor API usage limits

## 🎯 Integration with Kids Cartoon Generator

The voice synthesis system is fully integrated with the kids cartoon generator:

```python
# In kids_cartoon_generator.py
voice_provider = os.getenv("VOICE_PROVIDER", "coqui")

synthesizer = UnifiedVoiceSynthesizer(
    provider=voice_provider,
    config={
        "gpu": True,
        "speaker": "random",
        "text_temp": 0.8,  # Expressive for kids
        "waveform_temp": 0.7
    }
)

# Generate voice for kids content
audio_file = synthesizer.synthesize_voice(script["narration"], audio_path)
```

## 📚 References

- [Coqui TTS Bark Documentation](https://docs.coqui.ai/en/dev/models/bark.html)
- [ElevenLabs API Documentation](https://elevenlabs.io/docs)
- [Bark Model Paper](https://arxiv.org/abs/2307.04725)

## 🎉 Features Summary

✅ **Unified Interface**: Switch between providers seamlessly
✅ **Voice Cloning**: Clone voices from audio samples (Coqui TTS)
✅ **Kids Optimization**: Specialized settings for children's content
✅ **Offline Support**: Work without internet (Coqui TTS)
✅ **High Quality**: Professional-grade voice synthesis
✅ **Easy Integration**: Simple API for any application
✅ **Cost Effective**: Free local processing option
✅ **Privacy Focused**: Keep data local with Coqui TTS 