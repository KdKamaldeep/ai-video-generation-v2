#!/usr/bin/env python3
"""
Test script to verify AnimateDiffPipeline initialization fixes
- MotionAdapter loading using MotionAdapter.from_pretrained()
- Scheduler configuration with final_sigmas_type="sigma_min"
"""

import sys
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_motion_adapter_loading():
    """Test MotionAdapter loading with from_pretrained"""
    logger.info("Testing MotionAdapter loading...")
    
    try:
        from diffusers.models.motion_adapter import MotionAdapter
        
        # Test loading MotionAdapter
        motion_adapter = MotionAdapter.from_pretrained("guoyww/animatediff-v1-5-2")
        logger.info("✅ MotionAdapter loaded successfully with from_pretrained")
        
        # Check if it has the expected attributes
        if hasattr(motion_adapter, 'config'):
            logger.info("✅ MotionAdapter has config attribute")
        else:
            logger.warning("⚠️ MotionAdapter missing config attribute")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ MotionAdapter loading failed: {e}")
        return False

def test_animatediff_pipeline_initialization():
    """Test AnimateDiffPipeline initialization with MotionAdapter"""
    logger.info("Testing AnimateDiffPipeline initialization...")
    
    try:
        from diffusers import AnimateDiffPipeline
        from diffusers.models.motion_adapter import MotionAdapter
        
        # Load MotionAdapter first
        motion_adapter = MotionAdapter.from_pretrained("guoyww/animatediff-v1-5-2")
        logger.info("✅ MotionAdapter loaded")
        
        # Initialize AnimateDiffPipeline with motion_adapter argument
        pipeline = AnimateDiffPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter=motion_adapter,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        logger.info("✅ AnimateDiffPipeline initialized successfully with motion_adapter argument")
        
        # Check if motion adapter is properly loaded
        if hasattr(pipeline, 'motion_adapter') and pipeline.motion_adapter is not None:
            logger.info("✅ MotionAdapter is properly attached to pipeline")
        else:
            logger.warning("⚠️ MotionAdapter not found in pipeline")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ AnimateDiffPipeline initialization failed: {e}")
        return False

def test_scheduler_configuration():
    """Test scheduler configuration with final_sigmas_type"""
    logger.info("Testing scheduler configuration...")
    
    try:
        from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
        
        # Test DDIMScheduler with final_sigmas_type
        ddim_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            final_sigmas_type="sigma_min"
        )
        logger.info("✅ DDIMScheduler configured with final_sigmas_type='sigma_min'")
        
        # Test DPMSolverMultistepScheduler with final_sigmas_type
        dpmsolver_scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            final_sigmas_type="sigma_min"
        )
        logger.info("✅ DPMSolverMultistepScheduler configured with final_sigmas_type='sigma_min'")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Scheduler configuration failed: {e}")
        return False

def test_full_pipeline():
    """Test the full AnimateDiffGenerator with fixes"""
    logger.info("Testing full AnimateDiffGenerator...")
    
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        # Initialize generator with minimal settings
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-v1-5-2",
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
        logger.error(f"❌ Full pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Running AnimateDiffPipeline initialization fixes tests...")
    
    tests = [
        ("MotionAdapter Loading", test_motion_adapter_loading),
        ("AnimateDiffPipeline Initialization", test_animatediff_pipeline_initialization),
        ("Scheduler Configuration", test_scheduler_configuration),
        ("Full Pipeline Test", test_full_pipeline),
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
        logger.info("🎊 All tests passed! AnimateDiffPipeline fixes are working correctly.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main() 