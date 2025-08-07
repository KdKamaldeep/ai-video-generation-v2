#!/usr/bin/env python3
"""
Test script for AnimateDiff generate_animated_video_from_text function with S3 upload
"""

import os
import logging
import time
from typing import Dict, List, Optional
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Test scenarios
        test_scenarios = [
            {
                "name": "Basic Realistic Video",
                "text": "A beautiful sunset over the ocean with gentle waves",
                "style": "realistic",
                "num_frames": 80,  # 10 seconds at 8 FPS
                "fps": 8,
                "width": 512,
                "height": 768,
                "seed": 42
            },
            {
                "name": "Cartoon Style Video",
                "text": "A cute cartoon cat playing with a ball of yarn",
                "style": "cartoon",
                "num_frames": 120,  # 10 seconds at 12 FPS
                "fps": 12,
                "width": 512,
                "height": 768,
                "seed": 123
            },
            {
                "name": "Minimalist Style Video",
                "text": "Simple geometric shapes moving in a minimalist design",
                "style": "minimalist",
                "num_frames": 80,  # 10 seconds at 8 FPS
                "width": 512,
                "height": 768,
                "seed": 456
            },
            {
                "name": "Dramatic Style Video",
                "text": "A dramatic storm with lightning and dark clouds",
                "style": "dramatic",
                "num_frames": 100,  # 10 seconds at 10 FPS
                "fps": 10,
                "width": 512,
                "height": 768,
                "seed": 789
            },
            {
                "name": "Funny Style Video",
                "text": "A silly penguin slipping on ice and falling",
                "style": "funny",
                "num_frames": 80,  # 10 seconds at 8 FPS
                "fps": 8,
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
                
                # Generate video
                video_path = generator.generate_animated_video_from_text(
                    text=scenario['text'],
                    style=scenario['style'],
                    width=scenario['width'],
                    height=scenario['height'],
                    num_frames=scenario['num_frames'],
                    fps=scenario['fps'],
                    motion_strength=0.8,
                    num_inference_steps=20,
                    guidance_scale=7.5,
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
            "text": "A majestic eagle soaring through the clouds at sunset",
            "style": "realistic",
            "width": 512,
            "height": 768,
            "num_frames": 80,  # 10 seconds at 8 FPS
            "fps": 8,
            "motion_strength": 0.8,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "seed": 42
        }
        
        logger.info(f"Generating video for: {test_params['text']}")
        
        # Generate video
        start_time = time.time()
        video_path = generator.generate_animated_video_from_text(**test_params)
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
                logger.info(f"   Duration: ~{test_params['num_frames']/test_params['fps']:.1f} seconds")
                logger.info(f"   Frames: {test_params['num_frames']}")
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
    
    multi_test_result = test_animatediff_generator()
    
    # Test 2: Single video with S3 upload
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Single Video with S3 Upload")
    logger.info("="*60)
    
    single_test_result = test_single_video_with_upload()
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    
    if multi_test_result["success"]:
        summary = multi_test_result["summary"]
        logger.info(f"Multi-test Results:")
        logger.info(f"  Total: {summary['total']}")
        logger.info(f"  Successful: {summary['successful']}")
        logger.info(f"  Failed: {summary['failed']}")
        logger.info(f"  Success Rate: {summary['success_rate']:.1f}%")
        
        # Show S3 URLs for successful videos
        successful_results = [r for r in multi_test_result["results"] if r['status'] == '✅ SUCCESS']
        if successful_results:
            logger.info(f"\n🌐 S3 URLs for successful videos:")
            for result in successful_results:
                if result['s3_url']:
                    logger.info(f"  {result['scenario']}: {result['s3_url']}")
    
    if single_test_result["success"]:
        logger.info(f"Single Test Results:")
        logger.info(f"  Local Path: {single_test_result['local_path']}")
        logger.info(f"  S3 URL: {single_test_result['s3_url']}")
        logger.info(f"  File Size: {single_test_result['file_size_mb']:.2f} MB")
        logger.info(f"  Generation Time: {single_test_result['generation_time_seconds']:.2f} seconds")
    
    return {
        "multi_test": multi_test_result,
        "single_test": single_test_result
    }

if __name__ == "__main__":
    result = main()
    
    # Print final results for easy access
    print(f"\n📁 Final Results:")
    if result["multi_test"]["success"]:
        summary = result["multi_test"]["summary"]
        print(f"   Multi-test: {summary['successful']}/{summary['total']} successful ({summary['success_rate']:.1f}%)")
        
        # Print S3 URLs prominently
        successful_results = [r for r in result["multi_test"]["results"] if r['status'] == '✅ SUCCESS']
        if successful_results:
            print(f"\n🌐 S3 URLs:")
            for result_item in successful_results:
                if result_item['s3_url']:
                    print(f"   {result_item['scenario']}: {result_item['s3_url']}")
    
    if result["single_test"]["success"]:
        print(f"   Single test video: {result['single_test']['local_path']}")
        print(f"   Single test S3 URL: {result['single_test']['s3_url']}")
    else:
        print(f"   Single test failed: {result['single_test'].get('error', 'Unknown error')}") 