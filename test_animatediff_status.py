#!/usr/bin/env python3
"""
Test AnimateDiff status and provide fallback options
"""

import os
import sys
import logging

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from animatediff_generator import AnimateDiffGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_animatediff_status():
    """Test if AnimateDiff is working"""
    logger.info("🔍 Checking AnimateDiff Status...")
    
    try:
        # Initialize the generator
        generator = AnimateDiffGenerator()
        
        # Check pipeline status
        logger.info(f"✅ Stable Diffusion pipeline: {'Loaded' if generator.sd_pipeline else 'Failed'}")
        logger.info(f"✅ AnimateDiff pipeline: {'Loaded' if generator.animatediff_pipeline else 'Failed'}")
        
        if generator.animatediff_pipeline:
            logger.info("🎉 AnimateDiff is working! You can use motion features.")
            return True
        else:
            logger.warning("⚠️  AnimateDiff is not available. Motion features will be disabled.")
            logger.info("💡 You can still use Stable Diffusion for image generation.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking AnimateDiff status: {e}")
        return False

def test_image_generation_only():
    """Test image generation without AnimateDiff"""
    logger.info("🖼️  Testing Image Generation Only...")
    
    try:
        # Initialize the generator
        generator = AnimateDiffGenerator()
        
        # Test image generation
        logger.info("Generating test image...")
        image_path = generator.generate_image_from_text(
            text="A beautiful sunset over mountains",
            style="realistic",
            width=512,
            height=768,
            num_inference_steps=10,  # Reduced for faster testing
            guidance_scale=7.5,
            seed=42
        )
        
        if image_path:
            logger.info(f"✅ Image generated successfully: {image_path}")
            return True
        else:
            logger.error("❌ Image generation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in image generation test: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting AnimateDiff Status Test")
    
    # Test 1: Check AnimateDiff status
    animatediff_working = test_animatediff_status()
    
    # Test 2: Test image generation
    image_working = test_image_generation_only()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"✅ Stable Diffusion: {'Working' if image_working else 'Failed'}")
    logger.info(f"✅ AnimateDiff: {'Working' if animatediff_working else 'Not Available'}")
    
    if image_working and animatediff_working:
        logger.info("\n🎉 Everything is working! You can use:")
        logger.info("   • Text → Image generation")
        logger.info("   • Text → Video generation")
        logger.info("   • Image → Video (motion)")
        logger.info("   • S3 upload for both images and videos")
        
    elif image_working and not animatediff_working:
        logger.info("\n⚠️  Partial functionality available:")
        logger.info("   • ✅ Text → Image generation")
        logger.info("   • ❌ Text → Video generation (AnimateDiff not available)")
        logger.info("   • ❌ Image → Video (motion) (AnimateDiff not available)")
        logger.info("   • ✅ S3 upload for images")
        logger.info("\n💡 To enable video features:")
        logger.info("   • Install AnimateDiff: pip install diffusers[animatediff]")
        logger.info("   • Or use the image-only pipeline")
        
    else:
        logger.error("\n❌ Core functionality failed")
        logger.info("Please check your setup and dependencies")
    
    # Recommendations
    logger.info("\n💡 RECOMMENDATIONS:")
    if not animatediff_working:
        logger.info("   • Install AnimateDiff: pip install diffusers[animatediff]")
        logger.info("   • Or use the simple image generation pipeline")
        logger.info("   • Check your GPU and CUDA setup")
    
    if image_working:
        logger.info("   • You can still generate images and upload to S3")
        logger.info("   • Use simple_text_to_video.py for image-only pipeline")

if __name__ == "__main__":
    main() 