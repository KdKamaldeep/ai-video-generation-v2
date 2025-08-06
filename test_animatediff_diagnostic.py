#!/usr/bin/env python3
"""
Diagnostic test script for AnimateDiff integration
Helps identify and fix issues with AnimateDiff setup
"""

import os
import sys
import logging
import importlib

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    logger.info("=== Checking Dependencies ===")
    
    dependencies = {
        "torch": "PyTorch",
        "diffusers": "Diffusers library",
        "transformers": "Transformers library",
        "accelerate": "Accelerate library",
        "safetensors": "Safetensors library"
    }
    
    missing_deps = []
    available_deps = []
    
    for module, name in dependencies.items():
        try:
            importlib.import_module(module)
            available_deps.append(name)
            logger.info(f"✓ {name} available")
        except ImportError:
            missing_deps.append(name)
            logger.error(f"✗ {name} missing")
    
    # Check for AnimateDiff specifically
    try:
        from diffusers import AnimateDiffPipeline
        logger.info("✓ AnimateDiffPipeline available in diffusers")
        animatediff_available = True
    except ImportError:
        logger.error("✗ AnimateDiffPipeline not available in diffusers")
        logger.info("  Install with: pip install diffusers[animatediff]")
        animatediff_available = False
    
    return available_deps, missing_deps, animatediff_available

def check_cuda():
    """Check CUDA availability"""
    logger.info("=== Checking CUDA ===")
    
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            logger.warning("✗ CUDA not available - will use CPU")
            return False
    except Exception as e:
        logger.error(f"Error checking CUDA: {e}")
        return False

def test_model_access():
    """Test access to AnimateDiff models"""
    logger.info("=== Testing Model Access ===")
    
    models_to_test = [
        "ByteDance/AnimateDiff-v1-5",
        "ByteDance/AnimateDiff-v1-4", 
        "guoyww/animatediff-v1-5-2",
        "SG161222/Realistic_Vision_V5.1_noVAE"
    ]
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        for model_id in models_to_test:
            try:
                logger.info(f"Testing access to {model_id}...")
                # Try to get model info
                model_info = api.model_info(model_id)
                logger.info(f"✓ {model_id} accessible")
            except Exception as e:
                logger.error(f"✗ {model_id} not accessible: {e}")
                
    except ImportError:
        logger.warning("huggingface_hub not available - skipping model access test")
    except Exception as e:
        logger.error(f"Error testing model access: {e}")

def test_basic_import():
    """Test basic import of AnimateDiffGenerator"""
    logger.info("=== Testing Basic Import ===")
    
    try:
        from animatediff_generator import AnimateDiffGenerator
        logger.info("✓ AnimateDiffGenerator imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import AnimateDiffGenerator: {e}")
        return False

def test_initialization():
    """Test AnimateDiffGenerator initialization"""
    logger.info("=== Testing Initialization ===")
    
    try:
        from animatediff_generator import AnimateDiffGenerator
        
        logger.info("Initializing AnimateDiffGenerator...")
        generator = AnimateDiffGenerator()
        
        logger.info("✓ AnimateDiffGenerator initialized successfully")
        
        # Check pipeline status
        if generator.sd_pipeline is not None:
            logger.info("✓ Stable Diffusion pipeline loaded")
        else:
            logger.error("✗ Stable Diffusion pipeline failed to load")
        
        if generator.animatediff_pipeline is not None:
            logger.info("✓ AnimateDiff pipeline loaded")
        else:
            logger.warning("⚠ AnimateDiff pipeline not loaded (this might be expected)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize AnimateDiffGenerator: {e}")
        return False

def test_image_generation():
    """Test basic image generation"""
    logger.info("=== Testing Image Generation ===")
    
    try:
        from animatediff_generator import AnimateDiffGenerator
        
        generator = AnimateDiffGenerator()
        
        logger.info("Generating test image...")
        image_path = generator.generate_image_from_text(
            text="A simple test image",
            style="realistic",
            width=512,
            height=768,
            num_inference_steps=5,  # Very low for testing
            guidance_scale=7.5,
            seed=42
        )
        
        if image_path and os.path.exists(image_path):
            logger.info(f"✓ Image generated successfully: {image_path}")
            return True
        else:
            logger.error("✗ Image generation failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error in image generation test: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    logger.info("Starting AnimateDiff diagnostic tests...")
    
    # Test 1: Check dependencies
    available_deps, missing_deps, animatediff_available = check_dependencies()
    
    # Test 2: Check CUDA
    cuda_available = check_cuda()
    
    # Test 3: Test model access
    test_model_access()
    
    # Test 4: Test basic import
    import_success = test_basic_import()
    
    # Test 5: Test initialization
    init_success = False
    if import_success:
        init_success = test_initialization()
    
    # Test 6: Test image generation
    image_success = False
    if init_success:
        image_success = test_image_generation()
    
    # Summary
    logger.info("=== Diagnostic Summary ===")
    logger.info(f"Dependencies: {len(available_deps)} available, {len(missing_deps)} missing")
    logger.info(f"AnimateDiff available: {animatediff_available}")
    logger.info(f"CUDA available: {cuda_available}")
    logger.info(f"Import successful: {import_success}")
    logger.info(f"Initialization successful: {init_success}")
    logger.info(f"Image generation successful: {image_success}")
    
    if missing_deps:
        logger.info("\n=== Missing Dependencies ===")
        for dep in missing_deps:
            logger.info(f"Install: pip install {dep.lower()}")
    
    if not animatediff_available:
        logger.info("\n=== AnimateDiff Installation ===")
        logger.info("Install AnimateDiff support:")
        logger.info("pip install diffusers[animatediff]")
    
    if image_success:
        logger.info("\n🎉 Basic functionality working!")
    else:
        logger.info("\n❌ Issues detected. Check the logs above.")

if __name__ == "__main__":
    main() 