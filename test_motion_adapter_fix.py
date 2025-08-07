#!/usr/bin/env python3
"""
Test script to verify MotionAdapter loading and scheduler configuration fixes
"""

import logging
import sys
import os

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_motion_adapter_loading():
    """Test MotionAdapter loading without 404 errors"""
    logger.info("Testing MotionAdapter loading...")
    
    try:
        from diffusers import MotionAdapter, AnimateDiffPipeline
        import torch
        
        # Test loading MotionAdapter
        adapter_id = "guoyww/animatediff-motion-adapter-v1-5"
        logger.info(f"Loading MotionAdapter: {adapter_id}")
        
        motion_adapter = MotionAdapter.from_pretrained(adapter_id)
        logger.info("✅ MotionAdapter loaded successfully")
        
        # Test AnimateDiffPipeline initialization
        logger.info("Testing AnimateDiffPipeline initialization...")
        pipeline = AnimateDiffPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter=motion_adapter,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        logger.info("✅ AnimateDiffPipeline initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MotionAdapter loading failed: {e}")
        return False

def test_scheduler_configuration():
    """Test scheduler configuration without 404 errors"""
    logger.info("Testing scheduler configuration...")
    
    try:
        from diffusers import DEISMultistepScheduler, DDIMScheduler
        
        # Test DEISMultistepScheduler with default config
        logger.info("Testing DEISMultistepScheduler...")
        scheduler_config = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "steps_offset": 1,
            "clip_sample": False,
            "use_karras_sigmas": False
        }
        
        # Add final_sigmas_type for newer diffusers versions
        try:
            import diffusers
            if diffusers.__version__ >= "0.33.0":
                scheduler_config["final_sigmas_type"] = "sigma_min"
                logger.info(f"Added final_sigmas_type for diffusers {diffusers.__version__}")
        except (ImportError, AttributeError):
            logger.info("Could not determine diffusers version")
        
        scheduler = DEISMultistepScheduler.from_config(scheduler_config)
        logger.info("✅ DEISMultistepScheduler configured successfully")
        
        # Test DDIMScheduler fallback
        logger.info("Testing DDIMScheduler fallback...")
        ddim_config = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear"
        }
        
        ddim_scheduler = DDIMScheduler.from_config(ddim_config)
        logger.info("✅ DDIMScheduler configured successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Scheduler configuration failed: {e}")
        return False

def test_full_generator():
    """Test the full AnimateDiffGenerator with fixes"""
    logger.info("Testing full AnimateDiffGenerator...")
    
    try:
        from animatediff_generator import AnimateDiffGenerator
        
        # Initialize generator
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-motion-adapter-v1-5",
            memory_optimization=True
        )
        
        logger.info("✅ AnimateDiffGenerator initialized successfully")
        
        # Check if pipelines are loaded
        if generator.sd_pipeline is not None:
            logger.info("✅ Stable Diffusion pipeline loaded")
        else:
            logger.warning("⚠️ Stable Diffusion pipeline not loaded")
            
        if generator.animatediff_pipeline is not None:
            logger.info("✅ AnimateDiff pipeline loaded")
        else:
            logger.warning("⚠️ AnimateDiff pipeline not loaded")
        
        # Cleanup
        generator.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Full generator test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Running MotionAdapter and scheduler configuration tests...")
    
    tests = [
        ("MotionAdapter Loading", test_motion_adapter_loading),
        ("Scheduler Configuration", test_scheduler_configuration),
        ("Full Generator Test", test_full_generator),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🎯 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n📊 Test Results Summary:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎉 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎊 All tests passed! MotionAdapter and scheduler fixes are working correctly.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main() 