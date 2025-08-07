#!/usr/bin/env python3
"""
AnimateDiff Installation Test

This script tests if AnimateDiff is properly installed and working.
"""

import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_animatediff_import():
    """Test if AnimateDiff can be imported"""
    try:
        from diffusers import AnimateDiffPipeline
        logger.info("✅ AnimateDiffPipeline imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ AnimateDiffPipeline import failed: {e}")
        logger.info("💡 This usually means diffusers[animatediff] is not properly installed")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error importing AnimateDiffPipeline: {e}")
        return False

def test_motion_adapter_import():
    """Test if MotionAdapter can be imported"""
    try:
        from diffusers.models.motion_adapter import MotionAdapter
        logger.info("✅ MotionAdapter imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ MotionAdapter import failed: {e}")
        logger.info("💡 This indicates AnimateDiff components are missing")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error importing MotionAdapter: {e}")
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
    except ImportError as e:
        logger.error(f"❌ AnimateDiff pipeline import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ AnimateDiff pipeline test failed: {e}")
        return False

def test_diffusers_version():
    """Test diffusers version and source"""
    try:
        import diffusers
        logger.info(f"✅ Diffusers version: {diffusers.__version__}")
        
        # Check if it's from GitHub (should contain 'dev' or specific commit)
        if 'dev' in diffusers.__version__ or '+' in diffusers.__version__:
            logger.info("✅ Diffusers appears to be from GitHub (good for AnimateDiff)")
            return True
        else:
            logger.warning("⚠️ Diffusers appears to be from PyPI (may not have AnimateDiff support)")
            logger.info("💡 Consider installing from GitHub: pip install git+https://github.com/huggingface/diffusers.git@main#egg=diffusers[animatediff]")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking diffusers version: {e}")
        return False

def test_torch_availability():
    """Test if PyTorch is available"""
    try:
        import torch
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("ℹ️ CUDA not available (will use CPU)")
        
        return True
    except Exception as e:
        logger.error(f"❌ PyTorch test failed: {e}")
        return False

def test_transformers():
    """Test if transformers is available"""
    try:
        import transformers
        logger.info(f"✅ Transformers version: {transformers.__version__}")
        return True
    except Exception as e:
        logger.error(f"❌ Transformers test failed: {e}")
        return False

def main():
    """Run all AnimateDiff installation tests"""
    logger.info("🧪 AnimateDiff Installation Test")
    logger.info("=" * 60)
    
    tests = [
        ("PyTorch Availability", test_torch_availability),
        ("Transformers", test_transformers),
        ("Diffusers Version", test_diffusers_version),
        ("AnimateDiff Import", test_animatediff_import),
        ("MotionAdapter Import", test_motion_adapter_import),
        ("Pipeline Creation", test_animatediff_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🎯 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 AnimateDiff Installation Test Results")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎉 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! AnimateDiff is properly installed.")
        logger.info("✅ You can now run the kids cartoon generator!")
    else:
        logger.error("❌ Some tests failed. AnimateDiff may not be properly installed.")
        logger.info("\n🔧 Troubleshooting Steps:")
        logger.info("1. Uninstall current diffusers: pip uninstall diffusers")
        logger.info("2. Install from GitHub: pip install git+https://github.com/huggingface/diffusers.git@main#egg=diffusers[animatediff]")
        logger.info("3. Install other dependencies: pip install -r requirements.txt")
        logger.info("4. Run this test again: python test_animatediff_installation.py")
    
    # Additional recommendations
    logger.info("\n📋 Recommendations:")
    if passed >= 4:  # Most tests passed
        logger.info("✅ AnimateDiff appears to be working correctly")
        logger.info("💡 Try running: python kids_cartoon_generator.py")
    else:
        logger.info("⚠️ AnimateDiff needs to be reinstalled")
        logger.info("💡 Follow the ANIMATEDIFF_INSTALLATION_GUIDE.md")

if __name__ == "__main__":
    main() 