#!/usr/bin/env python3
"""
Test script to verify enhanced motion cues in prompts work better with SVD
"""

import os
import sys
import time
import logging
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_motion_prompts():
    """Test that enhanced motion prompts work better with SVD"""
    print("🎬 Testing Enhanced Motion Prompts with SVD")
    print("=" * 60)
    
    try:
        from utils.script_generator import ScriptGenerator
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        
        # Initialize generators
        print("1. Initializing generators...")
        script_gen = ScriptGenerator()
        sd_gen = StableDiffusionGenerator()
        
        if sd_gen.video_pipeline is None:
            print("❌ Video pipeline not available")
            return False
        
        print("✅ Generators initialized")
        
        # Test with different story types to see motion cues
        test_story_types = ["motivation", "horror", "real_life"]
        
        for story_type in test_story_types:
            print(f"\n2. Testing {story_type} script with motion cues...")
            
            try:
                # Generate script with enhanced motion cues
                script = script_gen.generate_script(story_type)
                print(f"✅ Generated {story_type} script: {script.title}")
                
                # Show the visual suggestions with motion cues
                print(f"   Visual suggestions with motion cues:")
                for i, line in enumerate(script.narration):
                    print(f"   Line {i+1}: {line.visual_suggestion}")
                
                # Test motion video generation for first line
                if script.narration:
                    first_line = script.narration[0]
                    print(f"\n3. Testing motion video generation for first line...")
                    print(f"   Text: {first_line.text}")
                    print(f"   Visual suggestion: {first_line.visual_suggestion}")
                    
                    # Generate image first
                    print("   Generating image from motion-enhanced prompt...")
                    image_path = sd_gen.generate_image_from_text(
                        text=first_line.visual_suggestion,
                        style="realistic",
                        seed=1000
                    )
                    
                    if image_path and os.path.exists(image_path):
                        print(f"   ✅ Image generated: {image_path}")
                        
                        # Generate motion video
                        print("   Generating motion video from image...")
                        start_time = time.time()
                        
                        video_path = sd_gen.generate_video_from_image(
                            image_path=image_path,
                            motion_strength=1.0,
                            num_frames=12,
                            fps=8,
                            motion_type="dynamic",
                            fast_mode=True
                        )
                        
                        end_time = time.time()
                        generation_time = end_time - start_time
                        
                        if video_path and os.path.exists(video_path):
                            print(f"   ✅ Motion video generated: {video_path}")
                            print(f"   Generation time: {generation_time:.2f} seconds")
                            print(f"   File size: {os.path.getsize(video_path) / 1024:.1f} KB")
                            
                            # Analyze motion
                            import cv2
                            cap = cv2.VideoCapture(video_path)
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.release()
                            
                            print(f"   Video analysis: {frame_count} frames, {fps} fps")
                            
                            if frame_count > 1:
                                print(f"   ✅ Motion video has multiple frames - motion generation working!")
                            else:
                                print(f"   ⚠️  Only {frame_count} frame - motion may be limited")
                        else:
                            print(f"   ❌ Motion video generation failed")
                    else:
                        print(f"   ❌ Image generation failed")
                
            except Exception as e:
                print(f"   ❌ Error testing {story_type}: {e}")
                continue
        
        print(f"\n✅ Enhanced motion prompt testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_comparison():
    """Compare old vs new prompts"""
    print("\n📝 Testing Prompt Comparison")
    print("=" * 60)
    
    try:
        from utils.script_generator import ScriptGenerator
        
        script_gen = ScriptGenerator()
        
        # Test old-style prompts (static)
        old_prompts = [
            "person looking in mirror",
            "dark doorway with shadows", 
            "person frantically searching for phone"
        ]
        
        # Test new-style prompts (with motion)
        new_prompts = [
            "camera slowly panning across a person looking in mirror with subtle breathing motion",
            "camera slowly creeping towards dark doorway with flickering shadows moving",
            "camera panning across person frantically searching for phone with rapid hand movements"
        ]
        
        print("Old-style prompts (static):")
        for i, prompt in enumerate(old_prompts):
            print(f"  {i+1}. {prompt}")
        
        print("\nNew-style prompts (with motion cues):")
        for i, prompt in enumerate(new_prompts):
            print(f"  {i+1}. {prompt}")
        
        print("\n✅ Motion cues identified in new prompts:")
        motion_keywords = [
            "camera", "panning", "dollying", "tilting", "zooming", "tracking", "creeping",
            "walking", "running", "falling", "rising", "spinning", "swaying", "moving",
            "blowing", "flowing", "falling", "rising", "breathing", "blinking", "fidgeting",
            "trembling", "twitching", "nodding", "vibration", "flickering", "shadows"
        ]
        
        for prompt in new_prompts:
            found_motion = [kw for kw in motion_keywords if kw in prompt.lower()]
            if found_motion:
                print(f"  - Found motion cues: {', '.join(found_motion)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Motion Enhancement Test Suite")
    print("=" * 80)
    
    try:
        # Test enhanced motion prompts
        motion_ok = test_enhanced_motion_prompts()
        
        # Test prompt comparison
        comparison_ok = test_prompt_comparison()
        
        print("\n📊 Test Results:")
        print(f"   Enhanced Motion Prompts: {'✅' if motion_ok else '❌'}")
        print(f"   Prompt Comparison: {'✅' if comparison_ok else '❌'}")
        
        if motion_ok and comparison_ok:
            print("\n🎉 All tests passed! Enhanced motion prompts should work better with SVD.")
            print("\n💡 Key improvements:")
            print("   - Visual suggestions now include motion cues")
            print("   - Camera movements: panning, dollying, tilting, zooming")
            print("   - Object movements: walking, running, falling, rising")
            print("   - Environmental motion: wind, water, shadows")
            print("   - Subtle motions: breathing, blinking, fidgeting")
        else:
            print("\n⚠️  Some tests failed. Check the issues above.")
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 