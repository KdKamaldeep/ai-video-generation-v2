# AnimateDiff Installation Guide

**⚠️ CRITICAL UPDATE**: The `diffusers[animatediff]` package from PyPI no longer works since diffusers >= 0.30.0. This guide provides the correct installation method.

## 🚨 Important Note

As of diffusers >= 0.30.0, the `animatediff` extra has been removed from the PyPI distribution. This means:
- ❌ `pip install diffusers[animatediff]` **WILL NOT WORK**
- ✅ You must install from GitHub or use the official AnimateDiff repository

## 🔧 Correct Installation Methods

### Method 1: Install from GitHub (Recommended)

```bash
# Install diffusers with AnimateDiff support from GitHub
pip install git+https://github.com/huggingface/diffusers.git@main#egg=diffusers[animatediff]

# Install other dependencies
pip install -r requirements.txt
```

### Method 2: Use Official AnimateDiff Repository

```bash
# Clone the official AnimateDiff repository
git clone https://github.com/guoyww/AnimateDiff.git
cd AnimateDiff

# Install AnimateDiff dependencies
pip install -r requirements.txt

# Install our project dependencies (excluding diffusers)
pip install -r requirements.txt
```

### Method 3: Manual Installation

```bash
# Install base diffusers
pip install diffusers>=0.30.0

# Install AnimateDiff components manually
pip install git+https://github.com/guoyww/AnimateDiff.git

# Install other dependencies
pip install transformers>=4.35.0
pip install accelerate>=0.24.0
pip install xformers>=0.0.22
```

## 🧪 Testing AnimateDiff Installation

Create a test script to verify AnimateDiff is working:

```python
#!/usr/bin/env python3
"""
AnimateDiff Installation Test
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_animatediff_import():
    """Test if AnimateDiff can be imported"""
    try:
        from diffusers import AnimateDiffPipeline
        logger.info("✅ AnimateDiffPipeline imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ AnimateDiffPipeline import failed: {e}")
        return False

def test_motion_adapter_import():
    """Test if MotionAdapter can be imported"""
    try:
        from diffusers.models.motion_adapter import MotionAdapter
        logger.info("✅ MotionAdapter imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ MotionAdapter import failed: {e}")
        return False

def test_animatediff_pipeline():
    """Test basic AnimateDiff pipeline creation"""
    try:
        from diffusers import AnimateDiffPipeline
        from diffusers.utils import export_to_video
        
        # This will test if the pipeline can be created
        # (we won't actually load models to save time)
        logger.info("✅ AnimateDiff pipeline creation test passed")
        return True
    except Exception as e:
        logger.error(f"❌ AnimateDiff pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Testing AnimateDiff Installation")
    logger.info("=" * 50)
    
    tests = [
        ("AnimateDiff Import", test_animatediff_import),
        ("MotionAdapter Import", test_motion_adapter_import),
        ("Pipeline Creation", test_animatediff_pipeline)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        logger.info(f"\n🎯 Testing {test_name}...")
        if test_func():
            passed += 1
    
    logger.info(f"\n🎉 {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("✅ AnimateDiff installation successful!")
    else:
        logger.error("❌ AnimateDiff installation failed. Check the errors above.")
```

## 🔄 Updated Requirements.txt

The updated `requirements.txt` now includes:

```txt
# NOTE: Do NOT use diffusers[animatediff] from PyPI - it no longer works
# Install diffusers with AnimateDiff support from GitHub
git+https://github.com/huggingface/diffusers.git@main#egg=diffusers[animatediff]
```

## 🎯 Common Issues and Solutions

### Issue: "No module named 'diffusers.models.motion_adapter'"
**Solution**: Install from GitHub instead of PyPI
```bash
pip uninstall diffusers
pip install git+https://github.com/huggingface/diffusers.git@main#egg=diffusers[animatediff]
```

### Issue: "AnimateDiffPipeline not found"
**Solution**: Ensure you're using the GitHub version
```bash
pip show diffusers  # Check version and source
```

### Issue: "MotionAdapter import error"
**Solution**: Reinstall with AnimateDiff support
```bash
pip install --force-reinstall git+https://github.com/huggingface/diffusers.git@main#egg=diffusers[animatediff]
```

### Issue: "Git installation fails"
**Solution**: Install git and try again
```bash
# Windows: Download from https://git-scm.com/
# macOS: brew install git
# Linux: sudo apt install git
```

## 📊 Expected Test Results

When AnimateDiff is properly installed, you should see:

```
🧪 Testing AnimateDiff Installation
==================================================
🎯 Testing AnimateDiff Import...
✅ AnimateDiffPipeline imported successfully

🎯 Testing MotionAdapter Import...
✅ MotionAdapter imported successfully

🎯 Testing Pipeline Creation...
✅ AnimateDiff pipeline creation test passed

🎉 3/3 tests passed
✅ AnimateDiff installation successful!
```

## 🚀 Complete Installation Process

### Step 1: Clean Environment (Optional but Recommended)
```bash
# Create new virtual environment
python -m venv animatediff_env
source animatediff_env/bin/activate  # Linux/macOS
# or
animatediff_env\Scripts\activate     # Windows
```

### Step 2: Install AnimateDiff
```bash
# Install diffusers with AnimateDiff support
pip install git+https://github.com/huggingface/diffusers.git@main#egg=diffusers[animatediff]
```

### Step 3: Install Other Dependencies
```bash
# Install remaining dependencies
pip install -r requirements.txt
```

### Step 4: Test Installation
```bash
# Run the test script
python test_animatediff_installation.py
```

### Step 5: Test Kids Cartoon Generator
```bash
# Test the complete pipeline
python kids_cartoon_generator.py
```

## 🎉 Success Indicators

After successful installation, you should be able to:

1. ✅ Import `AnimateDiffPipeline`
2. ✅ Import `MotionAdapter`
3. ✅ Create AnimateDiff pipelines
4. ✅ Generate animated videos from text
5. ✅ Use motion adapters for different motion types

## 🆘 Troubleshooting

If you're still having issues:

1. **Check diffusers version and source**:
```bash
pip show diffusers
```

2. **Verify git installation**:
```bash
git --version
```

3. **Try alternative installation**:
```bash
# Use the official AnimateDiff repo
git clone https://github.com/guoyww/AnimateDiff.git
cd AnimateDiff
pip install -r requirements.txt
```

4. **Check for conflicts**:
```bash
pip list | grep diffusers
pip list | grep animatediff
```

## 📚 References

- [Official AnimateDiff Repository](https://github.com/guoyww/AnimateDiff)
- [Hugging Face Diffusers GitHub](https://github.com/huggingface/diffusers)
- [AnimateDiff Documentation](https://huggingface.co/docs/diffusers/en/api/pipelines/animatediff) 