import os
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class S3Uploader:
    def __init__(self):
        self.access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket_name = os.getenv("S3_BUCKET_NAME", "why-would-you")
        
        if not self.access_key_id or not self.secret_access_key:
            logger.error("AWS credentials not found in environment variables")
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set")
        
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=self.region
            )
            logger.info(f"S3 client initialized for bucket: {self.bucket_name}")
        except NoCredentialsError:
            logger.error("AWS credentials are invalid")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
    
    def upload_audio_file(self, file_path: str, content_type: str = "audio/mpeg") -> Optional[str]:
        """
        Upload an audio file to S3 and return the public URL
        
        Args:
            file_path: Local path to the audio file
            content_type: MIME type of the audio file
            
        Returns:
            Public URL of the uploaded file, or None if upload failed
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return None
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = os.path.splitext(file_path)[1]
            unique_id = str(uuid.uuid4())[:8]
            s3_key = f"audio/{timestamp}_{unique_id}{file_extension}"
            
            logger.info(f"Uploading {file_path} to S3 as {s3_key}")
            
            # Upload file to S3
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'ACL': 'public-read'  # Make the file publicly accessible
                }
            )
            
            # Generate public URL
            public_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            logger.info(f"Successfully uploaded audio to S3: {public_url}")
            return public_url
            
        except ClientError as e:
            logger.error(f"AWS S3 error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading to S3: {e}")
            return None
    
    def upload_file_with_custom_key(self, file_path: str, s3_key: str, content_type: str = "audio/mpeg") -> Optional[str]:
        """
        Upload a file to S3 with a custom key and return the public URL
        
        Args:
            file_path: Local path to the file
            s3_key: Custom S3 key for the file
            content_type: MIME type of the file
            
        Returns:
            Public URL of the uploaded file, or None if upload failed
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return None
            
            logger.info(f"Uploading {file_path} to S3 as {s3_key}")
            
            # Upload file to S3
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'ACL': 'public-read'
                }
            )
            
            # Generate public URL
            public_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            logger.info(f"Successfully uploaded file to S3: {public_url}")
            return public_url
            
        except ClientError as e:
            logger.error(f"AWS S3 error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading to S3: {e}")
            return None
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3
        
        Args:
            s3_key: S3 key of the file to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Successfully deleted file from S3: {s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete file from S3: {e}")
            return False
    
    def list_files(self, prefix: str = "") -> list:
        """
        List files in the S3 bucket with optional prefix
        
        Args:
            prefix: Optional prefix to filter files
            
        Returns:
            List of file keys in the bucket
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            else:
                return []
                
        except ClientError as e:
            logger.error(f"Failed to list files from S3: {e}")
            return []
    
    def get_file_url(self, s3_key: str) -> str:
        """
        Get the public URL for a file in S3
        
        Args:
            s3_key: S3 key of the file
            
        Returns:
            Public URL of the file
        """
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}" 