#!/usr/bin/env python3
"""
Test script for AnimateDiff generate_animated_video_from_text function with S3 upload
Handles 24-frame limit by generating multiple chunks and looping them
"""

import os
import logging
import time
import subprocess
import tempfile
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_long_video_with_chunks(generator, text: str, style: str, target_duration: int = 10, 
                                  fps: int = 8, width: int = 512, height: int = 768, 
                                  seed: Optional[int] = None) -> Optional[str]:
    """
    Generate a longer video by creating multiple 24-frame chunks and looping them
    
    Args:
        generator: AnimateDiff generator instance
        text: Text prompt for video generation
        style: Video style
        target_duration: Target duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
        seed: Random seed
    
    Returns:
        Path to the final longer video, or None if failed
    """
    try:
        max_frames = 24  # AnimateDiff limit
        total_frames_needed = target_duration * fps
        num_chunks = (total_frames_needed + max_frames - 1) // max_frames  # Ceiling division
        
        logger.info(f"Generating {target_duration}s video at {fps} FPS")
        logger.info(f"Total frames needed: {total_frames_needed}")
        logger.info(f"Will generate {num_chunks} chunks of {max_frames} frames each")
        
        chunk_videos = []
        
        for chunk_idx in range(num_chunks):
            logger.info(f"Generating chunk {chunk_idx + 1}/{num_chunks}")
            
            # Generate chunk with slightly different seed for variety
            chunk_seed = seed + chunk_idx if seed is not None else None
            
            chunk_path = generator.generate_animated_video_from_text(
                text=text,
                style=style,
                width=width,
                height=height,
                num_frames=max_frames,
                fps=fps,
                motion_strength=0.8,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=chunk_seed
            )
            
            if chunk_path and os.path.exists(chunk_path):
                chunk_videos.append(chunk_path)
                logger.info(f"✅ Chunk {chunk_idx + 1} generated: {chunk_path}")
            else:
                logger.error(f"❌ Failed to generate chunk {chunk_idx + 1}")
                return None
        
        if not chunk_videos:
            logger.error("❌ No chunks were generated successfully")
            return None
        
        # Create longer video by concatenating chunks
        logger.info(f"Concatenating {len(chunk_videos)} chunks into longer video...")
        
        # Create file list for concatenation
        timestamp = int(time.time())
        file_list_path = os.path.join(os.path.dirname(chunk_videos[0]), f"chunk_list_{timestamp}.txt")
        
        with open(file_list_path, 'w') as f:
            for chunk_path in chunk_videos:
                f.write(f"file '{os.path.abspath(chunk_path)}'\n")
        
        # Generate final video path
        final_video_path = os.path.join(
            os.path.dirname(chunk_videos[0]), 
            f"long_video_{timestamp}.mp4"
        )
        
        # Concatenate chunks using FFmpeg
        try:
            subprocess.run([
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', file_list_path,
                '-framerate', str(fps),
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-vsync', 'cfr',
                '-avoid_negative_ts', 'make_zero',
                '-y', final_video_path
            ], check=True, capture_output=True)
            
            logger.info(f"✅ Long video created: {final_video_path}")
            
            # Clean up chunk files and file list
            os.remove(file_list_path)
            for chunk_path in chunk_videos:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            
            return final_video_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ FFmpeg concatenation failed: {e}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error generating long video: {e}")
        return None



def test_animatediff_generator():
    """Test the AnimateDiff generator with various scenarios and S3 upload"""
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        from utils.s3_uploader import S3Uploader
        
        logger.info("=" * 60)
        logger.info("Testing AnimateDiff Video Generation with S3 Upload")
        logger.info("=" * 60)
        
        # Initialize generator
        logger.info("Initializing AnimateDiff generator...")
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-motion-adapter-v1-5",
            memory_optimization=True
        )
        
        logger.info("✅ AnimateDiff generator initialized successfully")
        
        # Initialize S3 uploader
        logger.info("Initializing S3 uploader...")
        s3_uploader = S3Uploader()
        logger.info("✅ S3 uploader initialized successfully")
        
        # Test scenarios - now targeting 10+ seconds each
        test_scenarios = [
           {
                "name": "Penguin Comedy Story",
                "text": "A silly cartoon penguin slipping on ice and falling, colorful, comical scene",
                "style": "cartoon",         # Must match your style_enhancements keys
                "target_duration": 10,      # total video duration in seconds
                "fps": 8,                   # frames per second
                "width": 512,
                "height": 768,
                "seed": 321
            }

        ]
        
        results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            logger.info(f"\n{'='*40}")
            logger.info(f"Test {i}/{len(test_scenarios)}: {scenario['name']}")
            logger.info(f"{'='*40}")
            
            try:
                start_time = time.time()
                
                # Generate longer video using chunks
                video_path = generate_long_video_with_chunks(
                    generator=generator,
                    text=scenario['text'],
                    style=scenario['style'],
                    target_duration=scenario['target_duration'],
                    fps=scenario['fps'],
                    width=scenario['width'],
                    height=scenario['height'],
                    seed=scenario['seed']
                )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                if video_path and os.path.exists(video_path):
                    # Get file size
                    file_size = os.path.getsize(video_path)
                    file_size_mb = file_size / (1024 * 1024)
                    
                    # Upload to S3
                    logger.info("📤 Uploading to S3...")
                    timestamp = int(time.time())
                    filename = os.path.basename(video_path)
                    s3_key = f"animatediff-tests/{scenario['name'].lower().replace(' ', '-')}/{timestamp}_{filename}"
                    
                    s3_url = s3_uploader.upload_video(
                        local_file_path=video_path,
                        s3_key=s3_key,
                        folder="animatediff-tests"
                    )
                    
                    result = {
                        "scenario": scenario['name'],
                        "status": "✅ SUCCESS",
                        "video_path": video_path,
                        "s3_url": s3_url,
                        "file_size_mb": file_size_mb,
                        "generation_time_seconds": generation_time,
                        "parameters": scenario
                    }
                    
                    logger.info(f"✅ Video generated and uploaded successfully!")
                    logger.info(f"   Local path: {video_path}")
                    if s3_url:
                        logger.info(f"   S3 URL: {s3_url}")
                    else:
                        logger.warning("   S3 upload failed")
                    logger.info(f"   Size: {file_size_mb:.2f} MB")
                    logger.info(f"   Time: {generation_time:.2f} seconds")
                    logger.info(f"   Duration: {scenario['target_duration']} seconds")
                    
                else:
                    result = {
                        "scenario": scenario['name'],
                        "status": "❌ FAILED",
                        "video_path": None,
                        "s3_url": None,
                        "file_size_mb": 0,
                        "generation_time_seconds": generation_time,
                        "parameters": scenario,
                        "error": "Video generation failed"
                    }
                    
                    logger.error(f"❌ Video generation failed for scenario: {scenario['name']}")
                
                results.append(result)
                
                # Add delay between tests to avoid overwhelming the system
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error in scenario {scenario['name']}: {e}")
                result = {
                    "scenario": scenario['name'],
                    "status": "❌ ERROR",
                    "video_path": None,
                    "s3_url": None,
                    "file_size_mb": 0,
                    "generation_time_seconds": 0,
                    "parameters": scenario,
                    "error": str(e)
                }
                results.append(result)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        successful_tests = [r for r in results if r['status'] == '✅ SUCCESS']
        failed_tests = [r for r in results if r['status'] != '✅ SUCCESS']
        
        logger.info(f"Total tests: {len(results)}")
        logger.info(f"Successful: {len(successful_tests)}")
        logger.info(f"Failed: {len(failed_tests)}")
        logger.info(f"Success rate: {(len(successful_tests)/len(results)*100):.1f}%")
        
        if successful_tests:
            avg_time = sum(r['generation_time_seconds'] for r in successful_tests) / len(successful_tests)
            avg_size = sum(r['file_size_mb'] for r in successful_tests) / len(successful_tests)
            logger.info(f"Average generation time: {avg_time:.2f} seconds")
            logger.info(f"Average file size: {avg_size:.2f} MB")
        
        # Show successful videos with S3 URLs
        if successful_tests:
            logger.info(f"\n✅ SUCCESSFUL VIDEOS:")
            for result in successful_tests:
                logger.info(f"   {result['scenario']}:")
                logger.info(f"     Local: {result['video_path']}")
                if result['s3_url']:
                    logger.info(f"     S3: {result['s3_url']}")
                else:
                    logger.info(f"     S3: Upload failed")
        
        # Show failed tests
        if failed_tests:
            logger.info(f"\n❌ FAILED TESTS:")
            for result in failed_tests:
                logger.info(f"   {result['scenario']}: {result.get('error', 'Unknown error')}")
        
        # Cleanup
        generator.cleanup()
        
        return {
            "success": True,
            "results": results,
            "summary": {
                "total": len(results),
                "successful": len(successful_tests),
                "failed": len(failed_tests),
                "success_rate": len(successful_tests)/len(results)*100 if results else 0
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize AnimateDiff generator: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }

def test_single_video_with_upload():
    """Test a single video generation and upload to S3"""
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        from utils.s3_uploader import S3Uploader
        
        logger.info("=" * 60)
        logger.info("Testing Single Video Generation with S3 Upload")
        logger.info("=" * 60)
        
        # Initialize generator
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-motion-adapter-v1-5",
            memory_optimization=True
        )
        
        # Test parameters
        test_params = {
            "name": "Eagle Sunset Flight",
            "base_prompt": "A realistic eagle soaring through golden clouds at sunset, wings fully outstretched, sunlit feathers, detailed anatomy, flying smoothly through the sky, cinematic lighting, natural motion",
            "style": "realistic",
            "target_duration": 12,
            "fps": 12,
            "width": 768,
            "height": 512,
            "seed": 42
            }

        
        logger.info(f"Generating video for: {test_params['base_prompt']}")
        
        # Generate video using chunks
        start_time = time.time()
        video_path = generate_long_video_with_chunks(
            generator=generator,
            text=test_params['base_prompt'],
            style=test_params['style'],
            target_duration=test_params['target_duration'],
            fps=test_params['fps'],
            width=test_params['width'],
            height=test_params['height'],
            seed=test_params['seed']
        )
        generation_time = time.time() - start_time
        
        if video_path and os.path.exists(video_path):
            logger.info(f"✅ Video generated: {video_path}")
            logger.info(f"   Generation time: {generation_time:.2f} seconds")
            
            # Upload to S3
            logger.info("Uploading to S3...")
            s3_uploader = S3Uploader()
            
            # Generate S3 key
            timestamp = int(time.time())
            filename = os.path.basename(video_path)
            s3_key = f"animatediff-tests/single-video/{timestamp}_{filename}"
            
            s3_url = s3_uploader.upload_video(
                local_file_path=video_path,
                s3_key=s3_key,
                folder="animatediff-tests"
            )
            
            if s3_url:
                logger.info(f"✅ Video uploaded to S3: {s3_url}")
                
                # Get file info
                file_size = os.path.getsize(video_path)
                file_size_mb = file_size / (1024 * 1024)
                
                result = {
                    "success": True,
                    "local_path": video_path,
                    "s3_url": s3_url,
                    "file_size_mb": file_size_mb,
                    "generation_time_seconds": generation_time,
                    "parameters": test_params
                }
                
                logger.info(f"📊 Video Info:")
                logger.info(f"   Size: {file_size_mb:.2f} MB")
                logger.info(f"   Duration: {test_params['target_duration']} seconds")
                logger.info(f"   FPS: {test_params['fps']}")
                
                return result
            else:
                logger.error("❌ S3 upload failed")
                return {
                    "success": False,
                    "local_path": video_path,
                    "s3_url": None,
                    "error": "S3 upload failed"
                }
        else:
            logger.error("❌ Video generation failed")
            return {
                "success": False,
                "error": "Video generation failed"
            }
            
    except Exception as e:
        logger.error(f"❌ Error in single video test: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main test function"""
    logger.info("Starting AnimateDiff Video Generation Tests with S3 Upload")
    
    # Test 1: Multiple scenarios
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Multiple Video Generation Scenarios with S3 Upload")
    logger.info("="*60)
    
    #multi_test_result = test_animatediff_generator()
    
    # Test 2: Single video with S3 upload
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Single Video with S3 Upload")
    logger.info("="*60)
    
    single_test_result = test_single_video_with_upload()
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
   
    if single_test_result["success"]:
        logger.info(f"Single Test Results:")
        logger.info(f"  Local Path: {single_test_result['local_path']}")
        logger.info(f"  S3 URL: {single_test_result['s3_url']}")
        logger.info(f"  File Size: {single_test_result['file_size_mb']:.2f} MB")
        logger.info(f"  Generation Time: {single_test_result['generation_time_seconds']:.2f} seconds")
    
    return {
        "single_test": single_test_result
    }

if __name__ == "__main__":
    result = main()
    
    # Print final results for easy access
    print(f"\n📁 Final Results:")
    
    if result["single_test"]["success"]:
        print(f"   Single test video: {result['single_test']['local_path']}")
        print(f"   Single test S3 URL: {result['single_test']['s3_url']}")
    else:
        print(f"   Single test failed: {result['single_test'].get('error', 'Unknown error')}") 