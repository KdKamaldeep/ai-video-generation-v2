#!/usr/bin/env python3
"""
Test script to verify AnimateDiff adapter settings match the user's approach
"""

import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from diffusers.utils import export_to_gif
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_user_approach():
    """Test the exact approach provided by the user"""
    logger.info("Testing user's AnimateDiff approach...")
    
    try:
        # Load the motion adapter
        logger.info("Loading motion adapter...")
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5")
        logger.info("✅ Motion adapter loaded successfully")
        
        # load SD 1.5 based finetuned model
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        logger.info(f"Loading AnimateDiff pipeline with model: {model_id}")
        pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter)
        logger.info("✅ AnimateDiff pipeline loaded successfully")
        
        # Configure scheduler
        logger.info("Configuring DDIMScheduler...")
        scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1
        )
        pipe.scheduler = scheduler
        logger.info("✅ DDIMScheduler configured successfully")
        
        # Enable memory savings
        logger.info("Enabling memory optimizations...")
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
        logger.info("✅ Memory optimizations enabled")
        
        # Test generation
        logger.info("Testing video generation...")
        output = pipe(
            prompt=(
                "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
                "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
                "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
                "golden hour, coastal landscape, seaside scenery"
            ),
            negative_prompt="bad quality, worse quality",
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=torch.Generator("cpu").manual_seed(42),
        )
        frames = output.frames[0]
        
        logger.info(f"✅ Video generation successful! Generated {len(frames)} frames")
        
        # Save test video
        logger.info("Saving test video...")
        export_to_gif(frames, "test_user_approach.gif", fps=8)
        logger.info("✅ Test video saved as test_user_approach.gif")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_our_implementation():
    """Test our updated implementation"""
    logger.info("Testing our updated AnimateDiff implementation...")
    
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        
        # Initialize generator
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-motion-adapter-v1-5",
            memory_optimization=True
        )
        
        # Test video generation
        video_path = generator.generate_animated_video_from_text(
            text="masterpiece, bestquality, highlydetailed, ultradetailed, sunset, orange sky, warm lighting, fishing boats, ocean waves seagulls, rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, golden hour, coastal landscape, seaside scenery",
            style="realistic",
            num_frames=16,
            fps=8,
            num_inference_steps=25,
            guidance_scale=7.5,
            seed=42
        )
        
        if video_path:
            logger.info(f"✅ Our implementation successful! Video saved: {video_path}")
            return True
        else:
            logger.error("❌ Our implementation failed to generate video")
            return False
            
    except Exception as e:
        logger.error(f"❌ Our implementation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing AnimateDiff Adapter Settings")
    logger.info("=" * 60)
    
    # Test user's approach
    logger.info("\n1. Testing User's Approach:")
    user_success = test_user_approach()
    
    # Test our implementation
    logger.info("\n2. Testing Our Implementation:")
    our_success = test_our_implementation()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY:")
    logger.info(f"User's approach: {'✅ PASSED' if user_success else '❌ FAILED'}")
    logger.info(f"Our implementation: {'✅ PASSED' if our_success else '❌ FAILED'}")
    logger.info("=" * 60) 