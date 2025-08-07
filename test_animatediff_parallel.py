#!/usr/bin/env python3
"""
Parallel AnimateDiff Video Generation with Chunking
Generates 5 chunks (24 frames each) in parallel and stitches them into a single video
"""

import os
import logging
import time
import subprocess
import tempfile
import multiprocessing as mp
from typing import List, Tuple, Optional
from PIL import Image
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_animatediff_in_process(prompt: str, num_frames: int, seed: int) -> List[Image.Image]:
    """
    Generate AnimateDiff frames in a separate process.
    Each process creates its own pipeline to avoid CUDA sharing issues.
    
    Args:
        prompt: Text prompt for video generation
        num_frames: Number of frames to generate (max 24 for AnimateDiff)
        seed: Random seed for reproducibility
        
    Returns:
        List of PIL Image objects representing the video frames
    """
    try:
        from utils.animatediff_generator import AnimateDiffGenerator
        import torch
        
        # Create a new pipeline instance for this process
        logger.info(f"Creating pipeline for seed {seed}")
        generator = AnimateDiffGenerator(
            sd_model_id="SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter_id="guoyww/animatediff-motion-adapter-v1-5",
            memory_optimization=True
        )
        
        # Generate video frames
        result = generator.animatediff_pipeline(
            prompt=prompt,
            negative_prompt="bad quality, worse quality, low quality",
            width=512,
            height=768,
            num_frames=num_frames,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=torch.Generator(device=generator.device).manual_seed(seed)
        )
        
        # Return the frames
        frames = result.frames[0]
        logger.info(f"Generated {len(frames)} frames for seed {seed}")
        
        # Cleanup this process's pipeline
        generator.cleanup()
        
        return frames
        
    except Exception as e:
        logger.error(f"Error generating AnimateDiff frames for seed {seed}: {e}")
        return []

def generate_chunk_worker(args: Tuple[str, int, int, str]) -> Tuple[int, bool, str]:
    """
    Worker function to generate a single chunk in parallel.
    
    Args:
        args: Tuple of (prompt, chunk_index, seed, output_dir)
        
    Returns:
        Tuple of (chunk_index, success, chunk_dir)
    """
    prompt, chunk_index, seed, output_dir = args
    
    try:
        logger.info(f"Worker {chunk_index}: Starting chunk generation with seed {seed}")
        
        # Create chunk directory
        chunk_dir = os.path.join(output_dir, f"chunk_{chunk_index:02d}")
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Generate frames using process-specific pipeline
        frames = generate_animatediff_in_process(prompt, num_frames=24, seed=seed)
        
        if not frames:
            logger.error(f"Worker {chunk_index}: Failed to generate frames")
            return chunk_index, False, chunk_dir
        
        # Save frames to chunk directory
        for frame_idx, frame in enumerate(frames):
            frame_path = os.path.join(chunk_dir, f"frame_{frame_idx:04d}.png")
            frame.save(frame_path, "PNG")
        
        logger.info(f"Worker {chunk_index}: Saved {len(frames)} frames to {chunk_dir}")
        return chunk_index, True, chunk_dir
        
    except Exception as e:
        logger.error(f"Worker {chunk_index}: Error - {e}")
        return chunk_index, False, ""

def stitch_frames_to_video(chunk_dirs: List[str], output_video_path: str, fps: int = 8) -> bool:
    """
    Stitch all frames from chunk directories into a single video using ffmpeg.
    
    Args:
        chunk_dirs: List of chunk directory paths
        output_video_path: Path for the final video
        fps: Frames per second for the output video (default: 8 FPS)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Stitching {len(chunk_dirs)} chunks into video at {fps} FPS")
        
        # Create temporary directory for all frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_counter = 0
            
            # Copy all frames to temporary directory with sequential numbering
            for chunk_dir in chunk_dirs:
                if not os.path.exists(chunk_dir):
                    logger.error(f"Chunk directory not found: {chunk_dir}")
                    return False
                
                # Get all frame files in the chunk directory
                frame_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.png')])
                
                for frame_file in frame_files:
                    src_path = os.path.join(chunk_dir, frame_file)
                    dst_path = os.path.join(temp_dir, f"frame_{frame_counter:06d}.png")
                    shutil.copy2(src_path, dst_path)
                    frame_counter += 1
            
            logger.info(f"Copied {frame_counter} frames to temporary directory")
            
            # Use ffmpeg to create video from frames with high quality settings
            input_pattern = os.path.join(temp_dir, "frame_%06d.png")
            
            subprocess.run([
                'ffmpeg',
                '-framerate', str(fps),  # Set frame rate to 8 FPS
                '-i', input_pattern,
                '-c:v', 'libx264',  # Use H.264 codec
                '-preset', 'slow',  # Better quality than 'medium'
                '-crf', '18',  # Lower CRF for higher quality (18 is visually lossless)
                '-pix_fmt', 'yuv420p',  # Standard pixel format
                '-profile:v', 'high',  # High profile for better quality
                '-level', '4.1',  # Compatibility level
                '-movflags', '+faststart',  # Optimize for web streaming
                '-y', output_video_path
            ], check=True, capture_output=True)
            
            logger.info(f"✅ Video created: {output_video_path}")
            logger.info(f"   Frame rate: {fps} FPS")
            logger.info(f"   Quality: CRF 18 (high quality)")
            logger.info(f"   Preset: slow (better compression)")
            return True
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FFmpeg error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error stitching frames: {e}")
        return False

def generate_parallel_video(base_prompt: str = "A panda dancing in a bamboo forest", 
                          num_chunks: int = 5, 
                          base_seed: int = 42,
                          fps: int = 8,  # Changed default to 8 FPS
                          output_dir: str = "output/parallel_videos") -> Optional[str]:
    """
    Generate a video using parallel processing of multiple chunks.
    
    Args:
        base_prompt: Base prompt for video generation
        num_chunks: Number of chunks to generate (default: 5)
        base_seed: Base seed for generation (will be incremented for each chunk)
        fps: Frames per second for final video (default: 8 FPS)
        output_dir: Output directory for chunks and final video
        
    Returns:
        Path to the final video, or None if failed
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        
        logger.info("=" * 60)
        logger.info("Parallel AnimateDiff Video Generation")
        logger.info("=" * 60)
        logger.info(f"Base prompt: {base_prompt}")
        logger.info(f"Number of chunks: {num_chunks}")
        logger.info(f"Base seed: {base_seed}")
        logger.info(f"Target FPS: {fps}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("Note: Each process creates its own pipeline to avoid CUDA sharing issues")
        
        # Prepare arguments for parallel processing
        worker_args = []
        for chunk_idx in range(num_chunks):
            seed = base_seed + chunk_idx
            worker_args.append((base_prompt, chunk_idx, seed, output_dir))
        
        # Generate chunks in parallel
        logger.info(f"Starting parallel generation of {num_chunks} chunks...")
        start_time = time.time()
        
        # Use multiprocessing to generate chunks in parallel
        # Each process will create its own pipeline instance
        with mp.Pool(processes=min(num_chunks, mp.cpu_count())) as pool:
            results = pool.map(generate_chunk_worker, worker_args)
        
        generation_time = time.time() - start_time
        logger.info(f"Parallel generation completed in {generation_time:.2f} seconds")
        
        # Check results and collect successful chunk directories
        successful_chunks = []
        for chunk_index, success, chunk_dir in results:
            if success and os.path.exists(chunk_dir):
                successful_chunks.append(chunk_dir)
                logger.info(f"✅ Chunk {chunk_index}: Success")
            else:
                logger.error(f"❌ Chunk {chunk_index}: Failed")
        
        if not successful_chunks:
            logger.error("❌ No chunks were generated successfully")
            return None
        
        logger.info(f"Successfully generated {len(successful_chunks)} chunks")
        
        # Stitch frames into final video
        final_video_path = os.path.join(output_dir, f"parallel_video_{timestamp}.mp4")
        
        if stitch_frames_to_video(successful_chunks, final_video_path, fps):
            # Calculate video info
            total_frames = len(successful_chunks) * 24  # 24 frames per chunk
            duration = total_frames / fps
            
            logger.info(f"📊 Final Video Info:")
            logger.info(f"   Path: {final_video_path}")
            logger.info(f"   Duration: {duration:.1f} seconds")
            logger.info(f"   Total frames: {total_frames}")
            logger.info(f"   FPS: {fps}")
            logger.info(f"   Chunks: {len(successful_chunks)}")
            
            # Clean up chunk directories
            for chunk_dir in successful_chunks:
                shutil.rmtree(chunk_dir)
                logger.info(f"Cleaned up: {chunk_dir}")
            
            return final_video_path
        else:
            logger.error("❌ Failed to stitch frames into video")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error in parallel video generation: {e}")
        return None

def test_parallel_generation_with_s3():
    """Test parallel video generation and upload to S3"""
    try:
        from utils.s3_uploader import S3Uploader
        
        logger.info("=" * 60)
        logger.info("Testing Parallel Video Generation with S3 Upload")
        logger.info("=" * 60)
        
        # Generate video using parallel processing
        video_path = generate_parallel_video(
            base_prompt="A panda dancing in a bamboo forest",
            num_chunks=5,
            base_seed=42,
            fps=8
        )
        
        if not video_path or not os.path.exists(video_path):
            logger.error("❌ Video generation failed")
            return None
        
        # Upload to S3
        logger.info("📤 Uploading to S3...")
        s3_uploader = S3Uploader()
        
        timestamp = int(time.time())
        filename = os.path.basename(video_path)
        s3_key = f"animatediff-parallel/{timestamp}_{filename}"
        
        s3_url = s3_uploader.upload_video(
            local_file_path=video_path,
            s3_key=s3_key,
            folder="animatediff-parallel"
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
                "video_info": {
                    "duration": 15.0,  # 5 chunks * 24 frames / 8 FPS = 15 seconds
                    "total_frames": 120,
                    "chunks": 5,
                    "fps": 8
                }
            }
            
            logger.info(f"📊 Final Results:")
            logger.info(f"   Local path: {video_path}")
            logger.info(f"   S3 URL: {s3_url}")
            logger.info(f"   File size: {file_size_mb:.2f} MB")
            logger.info(f"   Duration: {result['video_info']['duration']} seconds")
            logger.info(f"   Total frames: {result['video_info']['total_frames']}")
            
            return result
        else:
            logger.error("❌ S3 upload failed")
            return {
                "success": False,
                "local_path": video_path,
                "s3_url": None,
                "error": "S3 upload failed"
            }
            
    except Exception as e:
        logger.error(f"❌ Error in parallel test: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main function"""
    logger.info("Starting Parallel AnimateDiff Video Generation")
    
    result = test_parallel_generation_with_s3()
    
    if result and result["success"]:
        logger.info("\n" + "=" * 60)
        logger.info("🎉 SUCCESS!")
        logger.info("=" * 60)
        logger.info(f"📁 Local video: {result['local_path']}")
        logger.info(f"🌐 S3 URL: {result['s3_url']}")
        logger.info(f"📊 File size: {result['file_size_mb']:.2f} MB")
        logger.info(f"🎬 Duration: {result['video_info']['duration']} seconds")
        logger.info(f"🎞️  Total frames: {result['video_info']['total_frames']}")
        logger.info(f"🔧 Chunks: {result['video_info']['chunks']}")
        
        # Print final S3 URL prominently
        print(f"\n" + "=" * 60)
        print(f"🎯 FINAL S3 URL:")
        print(f"{result['s3_url']}")
        print(f"=" * 60)
        
        return result
    else:
        logger.error("❌ Failed to generate and upload video")
        if result:
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
        return None

if __name__ == "__main__":
    main() 