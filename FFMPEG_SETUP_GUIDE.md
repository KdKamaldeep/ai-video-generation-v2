# FFmpeg Setup Guide

This guide helps you set up FFmpeg and related dependencies for the kids cartoon generator.

## 🚨 Current Issue

The error "No module named 'ffmpeg'" indicates that either:
1. FFmpeg binary is not installed on your system
2. ffmpeg-python package is not properly installed
3. There's a conflict between different video processing libraries

## 🔧 Quick Fix

### Step 1: Install FFmpeg Binary

#### Windows:
```bash
# Download from official site
# https://ffmpeg.org/download.html#build-windows

# Or use chocolatey
choco install ffmpeg

# Or use winget
winget install ffmpeg
```

#### macOS:
```bash
# Using Homebrew
brew install ffmpeg

# Or using MacPorts
sudo port install ffmpeg
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install ffmpeg
```

#### Linux (CentOS/RHEL):
```bash
sudo yum install ffmpeg
# or
sudo dnf install ffmpeg
```

### Step 2: Verify FFmpeg Installation

```bash
# Check if ffmpeg is in PATH
ffmpeg -version

# Should show something like:
# ffmpeg version 4.4.2 Copyright (c) 2000-2021 the FFmpeg developers
```

### Step 3: Install Python Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install video processing dependencies specifically
pip install ffmpeg-python==0.2.0
pip install opencv-python>=4.8.0
pip install moviepy==1.0.3
```

### Step 4: Test Setup

```bash
# Run the FFmpeg test script
python test_ffmpeg_setup.py
```

## 🧪 Testing Your Setup

Run the test script to verify everything is working:

```bash
python test_ffmpeg_setup.py
```

This will test:
- ✅ FFmpeg binary availability
- ✅ ffmpeg-python package
- ✅ MoviePy library
- ✅ OpenCV library
- ✅ Basic video processing

## 🔄 Alternative Solutions

### If FFmpeg Binary is Not Available

If you can't install FFmpeg system-wide, you can:

1. **Use MoviePy as fallback**:
```python
# The kids_cartoon_generator.py already has fallback support
# It will use alternative methods if ffmpeg-python is not available
```

2. **Use conda environment**:
```bash
conda install ffmpeg
conda install -c conda-forge ffmpeg-python
```

3. **Use Docker**:
```dockerfile
FROM python:3.9
RUN apt-get update && apt-get install -y ffmpeg
COPY requirements.txt .
RUN pip install -r requirements.txt
```

### If ffmpeg-python Conflicts

Try these alternatives:

```bash
# Uninstall and reinstall
pip uninstall ffmpeg-python
pip install ffmpeg-python==0.2.0

# Or try a different version
pip install ffmpeg-python==0.2.1
```

## 🎯 Common Issues and Solutions

### Issue: "No module named 'ffmpeg'"
**Solution**: Install ffmpeg-python package
```bash
pip install ffmpeg-python
```

### Issue: "ffmpeg command not found"
**Solution**: Install FFmpeg binary (see Step 1 above)

### Issue: "Permission denied" on Windows
**Solution**: Run as administrator or add FFmpeg to PATH manually

### Issue: "ImportError: No module named 'cv2'"
**Solution**: Install OpenCV
```bash
pip install opencv-python
```

### Issue: "MoviePy error"
**Solution**: Install MoviePy dependencies
```bash
pip install moviepy
pip install imageio-ffmpeg
```

## 📊 Expected Test Results

When everything is working correctly, you should see:

```
🧪 FFmpeg Setup Test
==================================================
🎯 Running FFmpeg Binary test...
✅ FFmpeg binary found and working
Version: 4.4.2

🎯 Running ffmpeg-python Package test...
✅ ffmpeg-python package imported successfully
✅ ffmpeg-python probe test passed

🎯 Running MoviePy test...
✅ moviepy imported successfully

🎯 Running OpenCV test...
✅ OpenCV imported successfully (version: 4.8.0)

🎯 Running Video Processing test...
✅ Video processing test successful

==================================================
📊 Test Results Summary
==================================================
FFmpeg Binary: ✅ PASS
ffmpeg-python Package: ✅ PASS
MoviePy: ✅ PASS
OpenCV: ✅ PASS
Video Processing: ✅ PASS

🎉 5/5 tests passed
🎉 All tests passed! FFmpeg setup is working correctly.
```

## 🎉 Success!

Once all tests pass, you can run the kids cartoon generator:

```bash
python kids_cartoon_generator.py
```

The generator will now work with proper video processing capabilities!

## 🆘 Still Having Issues?

If you're still experiencing problems:

1. **Check your Python environment**:
```bash
python --version
pip list | grep ffmpeg
```

2. **Verify FFmpeg installation**:
```bash
which ffmpeg  # Linux/macOS
where ffmpeg  # Windows
```

3. **Try a clean environment**:
```bash
# Create new virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

4. **Contact support** with your test results from `test_ffmpeg_setup.py` 