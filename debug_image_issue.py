#!/usr/bin/env python3
"""
Debug script to test image generation and identify the issue
"""

import os
import sys
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_to_s3(file_path: str, file_type: str = "image") -> str:
    """
    Upload generated file to S3
    
    Args:
        file_path: Path to the file to upload
        file_type: Type of file ("image" or "video")
        
    Returns:
        S3 URL of uploaded file or None if failed
    """
    try:
        from utils.s3_uploader import S3Uploader
        
        print(f"📤 Uploading {file_type} to S3...")
        uploader = S3Uploader()
        
        # Determine folder based on file type
        folder = "debug-images" if file_type == "image" else "debug-videos"
        
        # Upload file
        s3_url = uploader.upload_video(
            local_file_path=file_path,
            folder=folder
        )
        
        if s3_url:
            print(f"✅ {file_type.capitalize()} uploaded successfully: {s3_url}")
            return s3_url
        else:
            print(f"❌ Failed to upload {file_type} to S3")
            return None
            
    except Exception as e:
        print(f"❌ Error uploading {file_type} to S3: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_video_quality_parameters():
    """Test different video generation parameters to reduce distortion"""
    print("\n🎬 Testing Video Quality Parameters")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        
        # Initialize generator
        print("1. Initializing Stable Diffusion Generator...")
        generator = StableDiffusionGenerator()
        
        # First generate an image
        print("2. Generating test image...")
        test_prompt = "A person walking in a park on a sunny day"
        image_path = generator.generate_image_from_text(
            text=test_prompt,
            style="realistic"
        )
        
        if not image_path:
            print("❌ Image generation failed, cannot test video generation")
            return False
        
        print(f"✅ Image generated: {image_path}")
        
        # Test different parameter combinations
        test_configs = [
            {
                "name": "Conservative Settings",
                "motion_strength": 0.5,
                "noise_aug_strength": 0.1,
                "num_frames": 8,
                "fps": 8,
                "fast_mode": False,
                "num_inference_steps": 25
            },
            {
                "name": "Balanced Settings", 
                "motion_strength": 0.7,
                "noise_aug_strength": 0.2,
                "num_frames": 12,
                "fps": 8,
                "fast_mode": True,
                "num_inference_steps": 20
            },
            {
                "name": "High Quality Settings",
                "motion_strength": 0.8,
                "noise_aug_strength": 0.15,
                "num_frames": 16,
                "fps": 12,
                "fast_mode": False,
                "num_inference_steps": 30
            }
        ]
        
        results = []
        
        for i, config in enumerate(test_configs):
            print(f"\n3.{i+1} Testing {config['name']}...")
            print(f"   Motion strength: {config['motion_strength']}")
            print(f"   Noise augmentation: {config['noise_aug_strength']}")
            print(f"   Frames: {config['num_frames']}, FPS: {config['fps']}")
            print(f"   Inference steps: {config['num_inference_steps']}")
            
            start_time = time.time()
            video_path = generator.generate_video_from_image(
                image_path=image_path,
                motion_strength=config['motion_strength'],
                num_frames=config['num_frames'],
                fps=config['fps'],
                noise_aug_strength=config['noise_aug_strength'],
                fast_mode=config['fast_mode']
            )
            generation_time = time.time() - start_time
            
            if video_path:
                print(f"✅ Video generated: {video_path}")
                print(f"   Generation time: {generation_time:.2f} seconds")
                
                # Check file size
                if os.path.exists(video_path):
                    file_size = os.path.getsize(video_path)
                    print(f"   File size: {file_size} bytes")
                
                # Upload to S3 with descriptive name
                s3_url = upload_to_s3(video_path, f"video_{config['name'].lower().replace(' ', '_')}")
                if s3_url:
                    print(f"   S3 URL: {s3_url}")
                
                results.append({
                    "config": config['name'],
                    "success": True,
                    "path": video_path,
                    "s3_url": s3_url,
                    "generation_time": generation_time
                })
            else:
                print(f"❌ Video generation failed for {config['name']}")
                results.append({
                    "config": config['name'],
                    "success": False
                })
        
        # Print summary
        print("\n📊 Video Quality Test Results:")
        print("=" * 40)
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {result['config']}")
            if result['success']:
                print(f"   Time: {result['generation_time']:.2f}s")
                if result['s3_url']:
                    print(f"   S3: {result['s3_url']}")
        
        return len([r for r in results if r['success']]) > 0
        
    except Exception as e:
        print(f"❌ Error during video quality testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_generation():
    """Test image generation to see what's happening"""
    print("🔍 Testing Image Generation")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        
        # Initialize generator
        print("1. Initializing Stable Diffusion Generator...")
        generator = StableDiffusionGenerator()
        
        # Test image generation
        print("2. Generating test image...")
        test_prompt = "A person walking in a park on a sunny day"
        
        start_time = time.time()
        image_path = generator.generate_image_from_text(
            text=test_prompt,
            style="realistic"
        )
        generation_time = time.time() - start_time
        
        if image_path:
            print(f"✅ Image generated successfully: {image_path}")
            print(f"   Generation time: {generation_time:.2f} seconds")
            print(f"   File exists: {os.path.exists(image_path)}")
            print(f"   File extension: {os.path.splitext(image_path)[1]}")
            
            # Check file size
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                print(f"   File size: {file_size} bytes")
            
            # Upload to S3
            s3_url = upload_to_s3(image_path, "image")
            if s3_url:
                print(f"   S3 URL: {s3_url}")
            
            return True
        else:
            print("❌ Image generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during image generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_generation():
    """Test video generation from the generated image"""
    print("\n🎬 Testing Video Generation")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        
        # Initialize generator
        print("1. Initializing Stable Diffusion Generator...")
        generator = StableDiffusionGenerator()
        
        # First generate an image
        print("2. Generating test image...")
        test_prompt = "A person walking in a park on a sunny day"
        image_path = generator.generate_image_from_text(
            text=test_prompt,
            style="realistic"
        )
        
        if not image_path:
            print("❌ Image generation failed, cannot test video generation")
            return False
        
        print(f"✅ Image generated: {image_path}")
        
        # Test video generation
        print("3. Generating video from image...")
        start_time = time.time()
        video_path = generator.generate_video_from_image(
            image_path=image_path,
            motion_strength=1.0,
            num_frames=12,
            fps=8,
            fast_mode=True
        )
        generation_time = time.time() - start_time
        
        if video_path:
            print(f"✅ Video generated successfully: {video_path}")
            print(f"   Generation time: {generation_time:.2f} seconds")
            print(f"   File exists: {os.path.exists(video_path)}")
            print(f"   File extension: {os.path.splitext(video_path)[1]}")
            
            # Check file size
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                print(f"   File size: {file_size} bytes")
            
            # Upload to S3
            s3_url = upload_to_s3(video_path, "video")
            if s3_url:
                print(f"   S3 URL: {s3_url}")
            
            return True
        else:
            print("❌ Video generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during video generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_video_processing():
    """Test video generation with minimal processing to identify distortion source"""
    print("\n🔧 Testing Minimal Video Processing")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        import torch
        from diffusers.utils import export_to_video
        from PIL import Image
        import numpy as np
        import os
        import uuid
        from datetime import datetime
        
        # Initialize generator
        print("1. Initializing Stable Diffusion Generator...")
        generator = StableDiffusionGenerator()
        
        # First generate an image
        print("2. Generating test image...")
        test_prompt = "A person walking in a park on a sunny day"
        image_path = generator.generate_image_from_text(
            text=test_prompt,
            style="realistic"
        )
        
        if not image_path:
            print("❌ Image generation failed, cannot test video generation")
            return False
        
        print(f"✅ Image generated: {image_path}")
        
        # Test minimal processing approach
        print("3. Testing minimal video processing...")
        
        # Load image without preprocessing
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to standard video size
        image = image.resize((512, 288), Image.Resampling.LANCZOS)
        
        # Generate video with minimal settings
        pipeline_kwargs = {
            'decode_chunk_size': 8,
            'motion_bucket_id': 127,  # Conservative motion
            'fps': 8,
            'noise_aug_strength': 0.1,  # Very low noise
            'num_frames': 8,  # Fewer frames
            'num_inference_steps': 20,
        }
        
        print(f"   Using minimal settings: {pipeline_kwargs}")
        
        start_time = time.time()
        video_frames = generator.video_pipeline(image, **pipeline_kwargs).frames[0]
        generation_time = time.time() - start_time
        
        if video_frames:
            print(f"✅ Generated {len(video_frames)} frames in {generation_time:.2f}s")
            
            # Save with minimal processing
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"minimal_{timestamp}_{unique_id}.mp4"
            output_path = os.path.join(generator.video_output_dir, filename)
            
            # Convert frames to numpy arrays without color correction
            processed_frames = []
            for frame in video_frames:
                if hasattr(frame, 'size'):  # PIL Image
                    frame_array = np.array(frame)
                else:  # Already numpy array
                    frame_array = frame
                
                # Only basic normalization, no color correction
                if frame_array.dtype in [np.float16, np.float32, np.float64]:
                    if frame_array.max() <= 1.0:
                        frame_array = (frame_array * 255).astype(np.uint8)
                    else:
                        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
                elif frame_array.dtype != np.uint8:
                    frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
                
                processed_frames.append(frame_array)
            
            # Save video
            export_to_video(processed_frames, output_path, fps=8)
            
            if os.path.exists(output_path):
                print(f"✅ Minimal video saved: {output_path}")
                
                # Check file size
                file_size = os.path.getsize(output_path)
                print(f"   File size: {file_size} bytes")
                
                # Upload to S3
                s3_url = upload_to_s3(output_path, "video_minimal")
                if s3_url:
                    print(f"   S3 URL: {s3_url}")
                
                return True
            else:
                print("❌ Failed to save minimal video")
                return False
        else:
            print("❌ No frames generated")
            return False
            
    except Exception as e:
        print(f"❌ Error during minimal video testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_raw_video_generation():
    """Test completely raw video generation with zero processing"""
    print("\n🔬 Testing Raw Video Generation (Zero Processing)")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        import torch
        from diffusers.utils import export_to_video
        from PIL import Image
        import numpy as np
        import os
        import uuid
        from datetime import datetime
        
        # Initialize generator
        print("1. Initializing Stable Diffusion Generator...")
        generator = StableDiffusionGenerator()
        
        # First generate an image
        print("2. Generating test image...")
        test_prompt = "A person walking in a park on a sunny day"
        image_path = generator.generate_image_from_text(
            text=test_prompt,
            style="realistic"
        )
        
        if not image_path:
            print("❌ Image generation failed, cannot test video generation")
            return False
        
        print(f"✅ Image generated: {image_path}")
        
        # Test completely raw approach
        print("3. Testing raw video generation (zero processing)...")
        
        # Load image without ANY preprocessing
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Keep original size or use very small size
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
        
        # Use the most conservative settings possible
        pipeline_kwargs = {
            'decode_chunk_size': 1,  # Process one frame at a time
            'motion_bucket_id': 1,   # Minimal motion
            'fps': 4,                # Very low FPS
            'noise_aug_strength': 0.0,  # No noise augmentation
            'num_frames': 4,         # Very few frames
            'num_inference_steps': 10,  # Fewer steps
        }
        
        print(f"   Using ultra-conservative settings: {pipeline_kwargs}")
        
        start_time = time.time()
        video_frames = generator.video_pipeline(image, **pipeline_kwargs).frames[0]
        generation_time = time.time() - start_time
        
        if video_frames:
            print(f"✅ Generated {len(video_frames)} frames in {generation_time:.2f}s")
            
            # Save with absolutely no processing
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"raw_{timestamp}_{unique_id}.mp4"
            output_path = os.path.join(generator.video_output_dir, filename)
            
            # Convert frames with minimal processing
            processed_frames = []
            for i, frame in enumerate(video_frames):
                if hasattr(frame, 'size'):  # PIL Image
                    frame_array = np.array(frame)
                else:  # Already numpy array
                    frame_array = frame
                
                # Only basic type conversion, no color correction
                if frame_array.dtype in [np.float16, np.float32, np.float64]:
                    if frame_array.max() <= 1.0:
                        frame_array = (frame_array * 255).astype(np.uint8)
                    else:
                        frame_array = frame_array.astype(np.uint8)
                elif frame_array.dtype != np.uint8:
                    frame_array = frame_array.astype(np.uint8)
                
                # Ensure valid range without clipping
                if frame_array.min() < 0:
                    frame_array = frame_array - frame_array.min()
                if frame_array.max() > 255:
                    frame_array = (frame_array / frame_array.max() * 255).astype(np.uint8)
                
                processed_frames.append(frame_array)
                print(f"   Frame {i+1}: shape={frame_array.shape}, dtype={frame_array.dtype}, range=[{frame_array.min()}, {frame_array.max()}]")
            
            # Save video
            export_to_video(processed_frames, output_path, fps=4)
            
            if os.path.exists(output_path):
                print(f"✅ Raw video saved: {output_path}")
                
                # Check file size
                file_size = os.path.getsize(output_path)
                print(f"   File size: {file_size} bytes")
                
                # Upload to S3
                s3_url = upload_to_s3(output_path, "video_raw")
                if s3_url:
                    print(f"   S3 URL: {s3_url}")
                
                return True
            else:
                print("❌ Failed to save raw video")
                return False
        else:
            print("❌ No frames generated")
            return False
            
    except Exception as e:
        print(f"❌ Error during raw video testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_motion_buckets():
    """Test different motion bucket IDs to find the least distorted"""
    print("\n🎯 Testing Different Motion Bucket IDs")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        import torch
        from diffusers.utils import export_to_video
        from PIL import Image
        import numpy as np
        import os
        import uuid
        from datetime import datetime
        
        # Initialize generator
        print("1. Initializing Stable Diffusion Generator...")
        generator = StableDiffusionGenerator()
        
        # First generate an image
        print("2. Generating test image...")
        test_prompt = "A person walking in a park on a sunny day"
        image_path = generator.generate_image_from_text(
            text=test_prompt,
            style="realistic"
        )
        
        if not image_path:
            print("❌ Image generation failed, cannot test video generation")
            return False
        
        print(f"✅ Image generated: {image_path}")
        
        # Test different motion bucket IDs
        motion_buckets = [1, 2, 4, 8, 16, 32, 64, 127]
        
        results = []
        
        for bucket_id in motion_buckets:
            print(f"\n3. Testing motion bucket ID: {bucket_id}")
            
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((512, 288), Image.Resampling.LANCZOS)
            
            # Use conservative settings
            pipeline_kwargs = {
                'decode_chunk_size': 8,
                'motion_bucket_id': bucket_id,
                'fps': 8,
                'noise_aug_strength': 0.0,  # No noise
                'num_frames': 8,
                'num_inference_steps': 15,
            }
            
            start_time = time.time()
            video_frames = generator.video_pipeline(image, **pipeline_kwargs).frames[0]
            generation_time = time.time() - start_time
            
            if video_frames:
                print(f"✅ Generated {len(video_frames)} frames in {generation_time:.2f}s")
                
                # Save video
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"motion_bucket_{bucket_id}_{timestamp}_{unique_id}.mp4"
                output_path = os.path.join(generator.video_output_dir, filename)
                
                # Minimal processing
                processed_frames = []
                for frame in video_frames:
                    if hasattr(frame, 'size'):  # PIL Image
                        frame_array = np.array(frame)
                    else:  # Already numpy array
                        frame_array = frame
                    
                    # Basic normalization only
                    if frame_array.dtype in [np.float16, np.float32, np.float64]:
                        if frame_array.max() <= 1.0:
                            frame_array = (frame_array * 255).astype(np.uint8)
                        else:
                            frame_array = frame_array.astype(np.uint8)
                    elif frame_array.dtype != np.uint8:
                        frame_array = frame_array.astype(np.uint8)
                    
                    processed_frames.append(frame_array)
                
                # Save video
                export_to_video(processed_frames, output_path, fps=8)
                
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"   File size: {file_size} bytes")
                    
                    # Upload to S3
                    s3_url = upload_to_s3(output_path, f"video_motion_bucket_{bucket_id}")
                    if s3_url:
                        print(f"   S3 URL: {s3_url}")
                    
                    results.append({
                        "bucket_id": bucket_id,
                        "success": True,
                        "path": output_path,
                        "s3_url": s3_url,
                        "generation_time": generation_time
                    })
                else:
                    print(f"❌ Failed to save video for bucket {bucket_id}")
                    results.append({
                        "bucket_id": bucket_id,
                        "success": False
                    })
            else:
                print(f"❌ No frames generated for bucket {bucket_id}")
                results.append({
                    "bucket_id": bucket_id,
                    "success": False
                })
        
        # Print summary
        print("\n📊 Motion Bucket Test Results:")
        print("=" * 40)
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"{status} Motion Bucket {result['bucket_id']}")
            if result['success']:
                print(f"   Time: {result['generation_time']:.2f}s")
                if result['s3_url']:
                    print(f"   S3: {result['s3_url']}")
        
        return len([r for r in results if r['success']]) > 0
        
    except Exception as e:
        print(f"❌ Error during motion bucket testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_static_image_sequence():
    """Test generating a static image sequence (no motion) to avoid distortion"""
    print("\n🖼️ Testing Static Image Sequence")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        import torch
        from diffusers.utils import export_to_video
        from PIL import Image
        import numpy as np
        import os
        import uuid
        from datetime import datetime
        
        # Initialize generator
        print("1. Initializing Stable Diffusion Generator...")
        generator = StableDiffusionGenerator()
        
        # First generate an image
        print("2. Generating test image...")
        test_prompt = "A person walking in a park on a sunny day"
        image_path = generator.generate_image_from_text(
            text=test_prompt,
            style="realistic"
        )
        
        if not image_path:
            print("❌ Image generation failed, cannot test video generation")
            return False
        
        print(f"✅ Image generated: {image_path}")
        
        # Test static sequence approach
        print("3. Testing static image sequence...")
        
        # Load image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((512, 288), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Create a sequence of the same image (no motion)
        num_frames = 8
        static_frames = [image_array.copy() for _ in range(num_frames)]
        
        print(f"✅ Created {len(static_frames)} static frames")
        
        # Save as video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"static_{timestamp}_{unique_id}.mp4"
        output_path = os.path.join(generator.video_output_dir, filename)
        
        # Save video
        export_to_video(static_frames, output_path, fps=8)
        
        if os.path.exists(output_path):
            print(f"✅ Static video saved: {output_path}")
            
            # Check file size
            file_size = os.path.getsize(output_path)
            print(f"   File size: {file_size} bytes")
            
            # Upload to S3
            s3_url = upload_to_s3(output_path, "video_static")
            if s3_url:
                print(f"   S3 URL: {s3_url}")
            
            return True
        else:
            print("❌ Failed to save static video")
            return False
            
    except Exception as e:
        print(f"❌ Error during static video testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_very_slow_motion():
    """Test very slow motion generation to minimize distortion"""
    print("\n🐌 Testing Very Slow Motion")
    print("=" * 50)
    
    try:
        from utils.stable_diffusion_generator import StableDiffusionGenerator
        import torch
        from diffusers.utils import export_to_video
        from PIL import Image
        import numpy as np
        import os
        import uuid
        from datetime import datetime
        
        # Initialize generator
        print("1. Initializing Stable Diffusion Generator...")
        generator = StableDiffusionGenerator()
        
        # First generate an image
        print("2. Generating test image...")
        test_prompt = "A person walking in a park on a sunny day"
        image_path = generator.generate_image_from_text(
            text=test_prompt,
            style="realistic"
        )
        
        if not image_path:
            print("❌ Image generation failed, cannot test video generation")
            return False
        
        print(f"✅ Image generated: {image_path}")
        
        # Test very slow motion
        print("3. Testing very slow motion...")
        
        # Load image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((512, 288), Image.Resampling.LANCZOS)
        
        # Use very conservative settings for slow motion
        pipeline_kwargs = {
            'decode_chunk_size': 1,  # Process one frame at a time
            'motion_bucket_id': 1,   # Minimal motion
            'fps': 2,                # Very slow FPS
            'noise_aug_strength': 0.0,  # No noise
            'num_frames': 6,         # Few frames
            'num_inference_steps': 8,  # Very few steps
        }
        
        print(f"   Using very slow motion settings: {pipeline_kwargs}")
        
        start_time = time.time()
        video_frames = generator.video_pipeline(image, **pipeline_kwargs).frames[0]
        generation_time = time.time() - start_time
        
        if video_frames:
            print(f"✅ Generated {len(video_frames)} frames in {generation_time:.2f}s")
            
            # Save video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"slow_motion_{timestamp}_{unique_id}.mp4"
            output_path = os.path.join(generator.video_output_dir, filename)
            
            # Minimal processing
            processed_frames = []
            for frame in video_frames:
                if hasattr(frame, 'size'):  # PIL Image
                    frame_array = np.array(frame)
                else:  # Already numpy array
                    frame_array = frame
                
                # Basic normalization only
                if frame_array.dtype in [np.float16, np.float32, np.float64]:
                    if frame_array.max() <= 1.0:
                        frame_array = (frame_array * 255).astype(np.uint8)
                    else:
                        frame_array = frame_array.astype(np.uint8)
                elif frame_array.dtype != np.uint8:
                    frame_array = frame_array.astype(np.uint8)
                
                processed_frames.append(frame_array)
            
            # Save video
            export_to_video(processed_frames, output_path, fps=2)
            
            if os.path.exists(output_path):
                print(f"✅ Slow motion video saved: {output_path}")
                
                # Check file size
                file_size = os.path.getsize(output_path)
                print(f"   File size: {file_size} bytes")
                
                # Upload to S3
                s3_url = upload_to_s3(output_path, "video_slow_motion")
                if s3_url:
                    print(f"   S3 URL: {s3_url}")
                
                return True
            else:
                print("❌ Failed to save slow motion video")
                return False
        else:
            print("❌ No frames generated")
            return False
            
    except Exception as e:
        print(f"❌ Error during slow motion testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Debug Image Generation Issue")
    print("=" * 60)
    
    # Test image generation
    image_ok = test_image_generation()
    
    # Test minimal video processing (bypasses color correction)
    minimal_ok = test_minimal_video_processing()
    
    # Test video generation with quality parameters
    video_quality_ok = test_video_quality_parameters()
    
    # Test standard video generation
    video_ok = test_video_generation()
    
    # Test raw video generation (zero processing)
    raw_video_ok = test_raw_video_generation()

    # Test different motion bucket IDs
    motion_bucket_ok = test_different_motion_buckets()
    
    # Test static image sequence
    static_image_ok = test_static_image_sequence()

    # Test very slow motion
    very_slow_motion_ok = test_very_slow_motion()
    
    print("\n📊 Debug Results:")
    print(f"   Image Generation: {'✅' if image_ok else '❌'}")
    print(f"   Minimal Video Processing: {'✅' if minimal_ok else '❌'}")
    print(f"   Video Quality Tests: {'✅' if video_quality_ok else '❌'}")
    print(f"   Standard Video Generation: {'✅' if video_ok else '❌'}")
    print(f"   Raw Video Generation: {'✅' if raw_video_ok else '❌'}")
    print(f"   Motion Bucket Tests: {'✅' if motion_bucket_ok else '❌'}")
    print(f"   Static Image Sequence: {'✅' if static_image_ok else '❌'}")
    print(f"   Very Slow Motion: {'✅' if very_slow_motion_ok else '❌'}")
    
    if image_ok and (minimal_ok or video_quality_ok or video_ok or raw_video_ok or motion_bucket_ok or static_image_ok or very_slow_motion_ok):
        print("\n🎉 Generation working! Check the different video quality tests for best results.")
        if minimal_ok:
            print("💡 Try the minimal processing approach if videos are distorted!")
    else:
        print("\n⚠️  Issues detected - check logs above") 