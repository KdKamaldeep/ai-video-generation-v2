#!/usr/bin/env python3
"""
Quick Setup Test - Verify all dependencies and components

This script quickly tests if all required components are working:
- Python dependencies
- FFmpeg
- AWS credentials
- GPU availability
- Model access

Run this before the main tests to ensure everything is set up correctly.
"""

import os
import sys
import logging
import subprocess
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_python_dependencies() -> Dict[str, bool]:
    """Test if all required Python packages are installed"""
    logger.info("Testing Python dependencies...")
    
    required_packages = [
        'torch',
        'diffusers',
        'transformers',
        'ffmpeg-python',
        'boto3',
        'PIL',
        'numpy'
    ]
    
    results = {}
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
                results[package] = True
            else:
                __import__(package)
                results[package] = True
        except ImportError:
            results[package] = False
            logger.error(f"❌ Missing package: {package}")
    
    return results

def test_ffmpeg() -> bool:
    """Test if FFmpeg is available"""
    logger.info("Testing FFmpeg...")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ FFmpeg is available")
            return True
        else:
            logger.error("❌ FFmpeg command failed")
            return False
    except FileNotFoundError:
        logger.error("❌ FFmpeg not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.error("❌ FFmpeg command timed out")
        return False

def test_aws_credentials() -> bool:
    """Test if AWS credentials are configured"""
    logger.info("Testing AWS credentials...")
    
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
        
        # Try to create a session
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials:
            logger.info("✅ AWS credentials found")
            return True
        else:
            logger.error("❌ No AWS credentials found")
            return False
            
    except NoCredentialsError:
        logger.error("❌ AWS credentials not configured")
        return False
    except Exception as e:
        logger.error(f"❌ AWS test failed: {e}")
        return False

def test_gpu_availability() -> Dict[str, bool]:
    """Test GPU availability and CUDA"""
    logger.info("Testing GPU availability...")
    
    results = {}
    
    try:
        import torch
        
        # Test CUDA availability
        results['cuda_available'] = torch.cuda.is_available()
        if results['cuda_available']:
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            results['gpu_count'] = torch.cuda.device_count()
            logger.info(f"✅ GPU count: {results['gpu_count']}")
        else:
            logger.warning("⚠️  CUDA not available, will use CPU")
        
        # Test MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            results['mps_available'] = True
            logger.info("✅ MPS (Apple Silicon) available")
        else:
            results['mps_available'] = False
        
        return results
        
    except Exception as e:
        logger.error(f"❌ GPU test failed: {e}")
        return {'cuda_available': False, 'mps_available': False}

def test_model_access() -> Dict[str, bool]:
    """Test access to required models"""
    logger.info("Testing model access...")
    
    results = {}
    
    try:
        from diffusers import StableDiffusionPipeline
        
        # Test if we can access the model (without downloading)
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        logger.info(f"Testing access to {model_id}...")
        
        # This will fail if no internet, but that's expected
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=None,  # Don't load weights
                local_files_only=True  # Only check local cache
            )
            results['sd_model'] = True
            logger.info("✅ SD model available locally")
        except Exception:
            results['sd_model'] = False
            logger.warning("⚠️  SD model not cached locally (will download on first use)")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Model access test failed: {e}")
        return {'sd_model': False}

def test_s3_bucket_access() -> bool:
    """Test access to the S3 bucket"""
    logger.info("Testing S3 bucket access...")
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        s3 = boto3.client('s3')
        bucket_name = "why-would-you"
        
        # Try to list objects (minimal permission test)
        s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
        logger.info(f"✅ S3 bucket '{bucket_name}' accessible")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            logger.error(f"❌ S3 bucket 'why-would-you' does not exist")
        elif error_code == 'AccessDenied':
            logger.error(f"❌ Access denied to S3 bucket 'why-would-you'")
        else:
            logger.error(f"❌ S3 access error: {error_code}")
        return False
    except Exception as e:
        logger.error(f"❌ S3 test failed: {e}")
        return False

def main():
    """Run all setup tests"""
    logger.info("🔧 Quick Setup Test")
    logger.info("=" * 50)
    
    all_tests_passed = True
    results = {}
    
    # Test Python dependencies
    deps_results = test_python_dependencies()
    results['dependencies'] = deps_results
    if not all(deps_results.values()):
        all_tests_passed = False
    
    # Test FFmpeg
    ffmpeg_ok = test_ffmpeg()
    results['ffmpeg'] = ffmpeg_ok
    if not ffmpeg_ok:
        all_tests_passed = False
    
    # Test AWS credentials
    aws_ok = test_aws_credentials()
    results['aws'] = aws_ok
    if not aws_ok:
        all_tests_passed = False
    
    # Test GPU
    gpu_results = test_gpu_availability()
    results['gpu'] = gpu_results
    
    # Test model access
    model_results = test_model_access()
    results['models'] = model_results
    
    # Test S3 bucket access
    s3_ok = test_s3_bucket_access()
    results['s3_bucket'] = s3_ok
    if not s3_ok:
        all_tests_passed = False
    
    # Print summary
    logger.info("=" * 50)
    logger.info("📊 SETUP TEST SUMMARY")
    logger.info("=" * 50)
    
    print(f"Python Dependencies: {'✅' if all(deps_results.values()) else '❌'}")
    for pkg, status in deps_results.items():
        print(f"  - {pkg}: {'✅' if status else '❌'}")
    
    print(f"FFmpeg: {'✅' if ffmpeg_ok else '❌'}")
    print(f"AWS Credentials: {'✅' if aws_ok else '❌'}")
    print(f"S3 Bucket Access: {'✅' if s3_ok else '❌'}")
    
    print(f"GPU/CUDA: {'✅' if gpu_results.get('cuda_available', False) else '⚠️'}")
    if gpu_results.get('cuda_available'):
        print(f"  - GPU Count: {gpu_results.get('gpu_count', 0)}")
    
    print(f"Models: {'✅' if model_results.get('sd_model', False) else '⚠️'}")
    
    print("=" * 50)
    
    if all_tests_passed:
        logger.info("🎉 All critical tests passed! You can run the main tests.")
        logger.info("💡 Run: python simple_animatediff_test.py")
    else:
        logger.error("❌ Some tests failed. Please fix the issues above before running main tests.")
        logger.info("📖 Check ANIMATEDIFF_TEST_README.md for troubleshooting")
    
    return all_tests_passed

if __name__ == "__main__":
    main() 