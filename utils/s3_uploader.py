import boto3
import os
import logging
from typing import Optional
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

class S3Uploader:
    def __init__(self, bucket_name: str = "why-would-you", region: str = "us-east-1"):
        """
        Initialize S3 uploader
        
        Args:
            bucket_name: S3 bucket name
            region: AWS region
        """
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize S3 client with credentials"""
        try:
            # Try to get credentials from environment variables first
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            # If not in environment, try to read from creds.txt
            if not aws_access_key or not aws_secret_key:
                creds = self._read_credentials_from_file()
                if creds:
                    aws_access_key = creds.get('access-key')
                    aws_secret_key = creds.get('secret-key')
            
            if aws_access_key and aws_secret_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=self.region
                )
                logger.info(f"S3 client initialized for bucket: {self.bucket_name}")
            else:
                # Try to use default credentials (IAM roles, etc.)
                self.s3_client = boto3.client('s3', region_name=self.region)
                logger.info(f"S3 client initialized with default credentials for bucket: {self.bucket_name}")
                
        except NoCredentialsError:
            logger.error("No AWS credentials found")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
    
    def _read_credentials_from_file(self) -> Optional[dict]:
        """Read AWS credentials from creds.txt file"""
        try:
            creds_file = "creds.txt"
            if not os.path.exists(creds_file):
                return None
            
            credentials = {}
            with open(creds_file, 'r') as f:
                lines = f.readlines()
                
            current_section = None
            for line in lines:
                line = line.strip()
                if line == "aws":
                    current_section = "aws"
                elif current_section == "aws" and ":" in line:
                    key, value = line.split(":", 1)
                    credentials[key.strip()] = value.strip()
            
            return credentials if credentials else None
            
        except Exception as e:
            logger.error(f"Error reading credentials file: {e}")
            return None
    
    def upload_video(self, local_file_path: str, 
    s3_key: str = None, folder: str = "youtube-shorts", 
    content_type: str = "video/mp4") -> Optional[str]:
        """
        Upload a video file to S3
        
        Args:
            local_file_path: Path to local video file
            s3_key: Custom S3 key (optional, will generate if not provided)
            folder: S3 folder path (default: youtube-shorts)
            
        Returns:
            S3 URL of uploaded file or None if failed
        """
        try:
            if not os.path.exists(local_file_path):
                logger.error(f"Local file does not exist: {local_file_path}")
                return None
            
            # Generate S3 key if not provided
            if not s3_key:
                filename = os.path.basename(local_file_path)
                timestamp = os.path.getmtime(local_file_path)
                s3_key = f"{folder}/{timestamp}_{filename}"
            
            # Ensure folder prefix
            if not s3_key.startswith(folder):
                s3_key = f"{folder}/{s3_key}"
            
            logger.info(f"Uploading {local_file_path} to s3://{self.bucket_name}/{s3_key}")
            
            # Upload file
            self.s3_client.upload_file(
                local_file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'ACL': 'public-read'  # Make the video publicly accessible
                }
            )
            
            # Generate S3 URL
            s3_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"Video uploaded successfully: {s3_url}")
            
            return s3_url
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                logger.error(f"S3 bucket '{self.bucket_name}' does not exist")
            elif error_code == 'AccessDenied':
                logger.error("Access denied to S3 bucket. Check your credentials and permissions.")
            else:
                logger.error(f"S3 upload error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {e}")
            return None
    
    def list_videos(self, folder: str = "youtube-shorts", max_items: int = 50) -> list:
        """
        List videos in the specified S3 folder
        
        Args:
            folder: S3 folder path
            max_items: Maximum number of items to return
            
        Returns:
            List of video objects
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=folder,
                MaxKeys=max_items
            )
            
            videos = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        videos.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'url': f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{obj['Key']}"
                        })
            
            logger.info(f"Found {len(videos)} videos in {folder}")
            return videos
            
        except Exception as e:
            logger.error(f"Error listing videos: {e}")
            return []
    
    def delete_video(self, s3_key: str) -> bool:
        """
        Delete a video from S3
        
        Args:
            s3_key: S3 key of the video to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted video: s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting video: {e}")
            return False
    
    def get_video_url(self, s3_key: str) -> str:
        """
        Generate S3 URL for a video
        
        Args:
            s3_key: S3 key of the video
            
        Returns:
            S3 URL
        """
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}" 