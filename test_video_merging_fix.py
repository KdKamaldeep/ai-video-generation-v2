#!/usr/bin/env python3
"""
Test script to verify the video merging fixes work correctly and upload to S3
"""

import os
import subprocess
import tempfile
import logging
from typing import List, Optional
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_video(duration_seconds: int, fps: int, output_path: str, frame_pattern: str = "test") -> str:
    """Create a test video with specified duration and frame rate"""
    try:
        # Create a simple test video using FFmpeg
        subprocess.run([
            'ffmpeg', '-f', 'lavfi',
            '-i', f'testsrc=duration={duration_seconds}:size=512x512:rate={fps}',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-y', output_path
        ], check=True, capture_output=True)
        
        logger.info(f"✅ Created test video: {output_path} ({duration_seconds}s, {fps} FPS)")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Failed to create test video: {e}")
        return None

def get_video_info(video_path: str) -> dict:
    """Get video information using FFmpeg probe"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ], capture_output=True, text=True, check=True)
        
        import json
        info = json.loads(result.stdout)
        
        # Extract video stream info
        video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
        if video_stream:
            duration = float(info['format']['duration'])
            fps = eval(video_stream['r_frame_rate'])  # e.g., "8/1" -> 8.0
            frame_count = int(duration * fps)
            
            return {
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'width': int(video_stream['width']),
                'height': int(video_stream['height'])
            }
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Failed to get video info: {e}")
        return None

def upload_to_s3(video_path: str, folder: str = "video-merging-tests") -> Optional[str]:
    """Upload video to S3 using the existing S3Uploader"""
    try:
        from utils.s3_uploader import S3Uploader
        
        # Initialize S3 uploader
        s3_uploader = S3Uploader()
        
        # Generate S3 key with timestamp
        timestamp = int(time.time())
        filename = os.path.basename(video_path)
        s3_key = f"{folder}/{timestamp}_{filename}"
        
        # Upload to S3
        s3_url = s3_uploader.upload_video(
            local_file_path=video_path,
            s3_key=s3_key,
            folder=folder
        )
        
        if s3_url:
            logger.info(f"✅ Video uploaded to S3: {s3_url}")
            return s3_url
        else:
            logger.error("❌ Failed to upload video to S3")
            return None
            
    except Exception as e:
        logger.error(f"❌ S3 upload error: {e}")
        return None

def test_old_merging_method(video_paths: List[str], output_path: str) -> bool:
    """Test the old merging method (with issues)"""
    try:
        logger.info("Testing OLD merging method (with known issues)...")
        
        # Create file list
        file_list_path = os.path.join(os.path.dirname(output_path), "old_video_list.txt")
        with open(file_list_path, 'w') as f:
            for video_path in video_paths:
                f.write(f"file '{os.path.abspath(video_path)}'\n")
        
        # Old method with conflicting filters
        subprocess.run([
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', file_list_path,
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-r', '8', '-pix_fmt', 'yuv420p',
            '-vf', 'fps=8:round=up',  # This causes issues
            '-y', output_path
        ], check=True, capture_output=True)
        
        # Clean up
        os.remove(file_list_path)
        
        return os.path.exists(output_path)
        
    except Exception as e:
        logger.error(f"❌ Old merging method failed: {e}")
        return False

def test_new_merging_method(video_paths: List[str], output_path: str) -> bool:
    """Test the new merging method (fixed)"""
    try:
        logger.info("Testing NEW merging method (fixed)...")
        
        # Step 1: Normalize all input videos to 8 FPS first
        normalized_video_paths = []
        for i, video_path in enumerate(video_paths):
            normalized_path = os.path.join(os.path.dirname(output_path), f"normalized_{i}.mp4")
            logger.info(f"Normalizing video {i+1}/{len(video_paths)} to 8 FPS")
            
            # Re-encode each video to exactly 8 FPS with proper PTS
            subprocess.run([
                'ffmpeg', '-i', video_path,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-r', '8', '-pix_fmt', 'yuv420p',
                '-vsync', 'cfr',  # Constant frame rate
                '-y', normalized_path
            ], check=True, capture_output=True)
            
            normalized_video_paths.append(normalized_path)
        
        # Step 2: Create file list for video concatenation
        file_list_path = os.path.join(os.path.dirname(output_path), "new_video_list.txt")
        with open(file_list_path, 'w') as f:
            for video_path in normalized_video_paths:
                f.write(f"file '{os.path.abspath(video_path)}'\n")
        
        # Step 3: Combine videos with proper PTS handling
        subprocess.run([
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', file_list_path,
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-r', '8', '-pix_fmt', 'yuv420p',
            '-vsync', 'cfr',  # Constant frame rate
            '-avoid_negative_ts', 'make_zero',  # Handle PTS properly
            '-y', output_path
        ], check=True, capture_output=True)
        
        # Clean up temporary files
        os.remove(file_list_path)
        for normalized_path in normalized_video_paths:
            if os.path.exists(normalized_path):
                os.remove(normalized_path)
        
        return os.path.exists(output_path)
        
    except Exception as e:
        logger.error(f"❌ New merging method failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("Testing Video Merging Fixes with S3 Upload")
    logger.info("=" * 60)
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory: {temp_dir}")
        
        # Create test videos (5 videos, each ~10 seconds, 8 FPS)
        test_videos = []
        for i in range(5):
            video_path = os.path.join(temp_dir, f"test_video_{i}.mp4")
            duration = 10 + (i * 0.5)  # 10, 10.5, 11, 11.5, 12 seconds
            
            if create_test_video(duration, 8, video_path, f"test_{i}"):
                test_videos.append(video_path)
        
        if len(test_videos) != 5:
            logger.error("❌ Failed to create all test videos")
            return None
        
        logger.info(f"✅ Created {len(test_videos)} test videos")
        
        # Calculate expected total duration and frames
        total_expected_duration = sum(get_video_info(video)['duration'] for video in test_videos)
        total_expected_frames = sum(get_video_info(video)['frame_count'] for video in test_videos)
        
        logger.info(f"Expected total duration: {total_expected_duration:.2f} seconds")
        logger.info(f"Expected total frames: {total_expected_frames}")
        
        # Test old method
        old_output = os.path.join(temp_dir, "old_merged.mp4")
        old_success = test_old_merging_method(test_videos, old_output)
        
        old_s3_url = None
        if old_success:
            old_info = get_video_info(old_output)
            if old_info:
                logger.info(f"OLD method result:")
                logger.info(f"  Duration: {old_info['duration']:.2f}s (expected: {total_expected_duration:.2f}s)")
                logger.info(f"  Frames: {old_info['frame_count']} (expected: {total_expected_frames})")
                logger.info(f"  FPS: {old_info['fps']}")
                
                duration_diff = abs(old_info['duration'] - total_expected_duration)
                frame_diff = abs(old_info['frame_count'] - total_expected_frames)
                
                if duration_diff > 1.0 or frame_diff > 10:
                    logger.warning("⚠️ OLD method has significant duration/frame loss (expected)")
                else:
                    logger.info("✅ OLD method worked correctly (unexpected)")
            
            # Upload old method video to S3
            logger.info("Uploading OLD method video to S3...")
            old_s3_url = upload_to_s3(old_output, "video-merging-tests/old-method")
        
        # Test new method
        new_output = os.path.join(temp_dir, "new_merged.mp4")
        new_success = test_new_merging_method(test_videos, new_output)
        
        new_s3_url = None
        if new_success:
            new_info = get_video_info(new_output)
            if new_info:
                logger.info(f"NEW method result:")
                logger.info(f"  Duration: {new_info['duration']:.2f}s (expected: {total_expected_duration:.2f}s)")
                logger.info(f"  Frames: {new_info['frame_count']} (expected: {total_expected_frames})")
                logger.info(f"  FPS: {new_info['fps']}")
                
                duration_diff = abs(new_info['duration'] - total_expected_duration)
                frame_diff = abs(new_info['frame_count'] - total_expected_frames)
                
                if duration_diff <= 0.5 and frame_diff <= 5:
                    logger.info("✅ NEW method preserved duration and frames correctly!")
                else:
                    logger.error(f"❌ NEW method still has issues: duration diff={duration_diff:.2f}s, frame diff={frame_diff}")
            
            # Upload new method video to S3
            logger.info("Uploading NEW method video to S3...")
            new_s3_url = upload_to_s3(new_output, "video-merging-tests/new-method")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY:")
        logger.info(f"Old method: {'✅ WORKED' if old_success else '❌ FAILED'}")
        logger.info(f"New method: {'✅ WORKED' if new_success else '❌ FAILED'}")
        logger.info("=" * 60)
        
        # Return the final video paths and S3 URLs
        final_results = {
            'old_method': {
                'local_path': old_output if old_success else None,
                's3_url': old_s3_url
            },
            'new_method': {
                'local_path': new_output if new_success else None,
                's3_url': new_s3_url
            },
            'temp_dir': temp_dir,
            'test_videos': test_videos
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS:")
        if old_success:
            logger.info(f"OLD method video: {old_output}")
            if old_s3_url:
                logger.info(f"OLD method S3 URL: {old_s3_url}")
        if new_success:
            logger.info(f"NEW method video: {new_output}")
            if new_s3_url:
                logger.info(f"NEW method S3 URL: {new_s3_url}")
        logger.info(f"Temporary directory: {temp_dir}")
        logger.info("=" * 60)
        
        return final_results

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n📁 Final Results:")
        if result['old_method']['local_path']:
            print(f"   Old method video: {result['old_method']['local_path']}")
            if result['old_method']['s3_url']:
                print(f"   Old method S3 URL: {result['old_method']['s3_url']}")
        if result['new_method']['local_path']:
            print(f"   New method video: {result['new_method']['local_path']}")
            if result['new_method']['s3_url']:
                print(f"   New method S3 URL: {result['new_method']['s3_url']}")
        print(f"   Temp directory: {result['temp_dir']}")
        print(f"   Test videos: {len(result['test_videos'])} videos created")
    else:
        print("❌ Test failed") 