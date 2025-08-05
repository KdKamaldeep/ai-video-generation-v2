import os
import json
from typing import Optional, Dict, Any
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
import pickle

class YouTubeUploader:
    def __init__(self):
        self.client_id = os.getenv("YOUTUBE_CLIENT_ID")
        self.client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
        self.redirect_uri = os.getenv("YOUTUBE_REDIRECT_URI", "http://localhost:8000/auth/callback")
        self.scopes = ['https://www.googleapis.com/auth/youtube.upload']
        self.credentials = None
        self.youtube = None
    
    def authenticate(self) -> bool:
        """Authenticate with YouTube API using OAuth2"""
        try:
            # Check if we have stored credentials
            if os.path.exists('token.pickle'):
                with open('token.pickle', 'rb') as token:
                    self.credentials = pickle.load(token)
            
            # If credentials are invalid or don't exist, refresh or get new ones
            if not self.credentials or not self.credentials.valid:
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    self.credentials.refresh(Request())
                else:
                    # Create flow for OAuth2
                    flow = InstalledAppFlow.from_client_config(
                        {
                            "installed": {
                                "client_id": self.client_id,
                                "client_secret": self.client_secret,
                                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                "token_uri": "https://oauth2.googleapis.com/token",
                                "redirect_uris": [self.redirect_uri]
                            }
                        },
                        self.scopes
                    )
                    
                    # Run the OAuth2 flow
                    self.credentials = flow.run_local_server(port=8080)
                
                # Save credentials for next run
                with open('token.pickle', 'wb') as token:
                    pickle.dump(self.credentials, token)
            
            # Build the YouTube service
            self.youtube = build('youtube', 'v3', credentials=self.credentials)
            return True
            
        except Exception as e:
            print(f"Authentication failed: {e}")
            return False
    
    def upload_video(self, video_path: str, title: str, description: str, 
                    tags: list, category_id: str = "22") -> Optional[Dict[str, Any]]:
        """Upload video to YouTube"""
        
        if not self.authenticate():
            return None
        
        try:
            # Prepare the video metadata
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': tags,
                    'categoryId': category_id,
                    'defaultLanguage': 'en',
                    'defaultAudioLanguage': 'en'
                },
                'status': {
                    'privacyStatus': 'private',  # Start as private for safety
                    'selfDeclaredMadeForKids': False
                }
            }
            
            # Create the media upload
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
            
            # Upload the video
            request = self.youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    print(f"Uploaded {int(status.progress() * 100)}%")
            
            print(f"Upload Complete! Video ID: {response['id']}")
            return {
                'video_id': response['id'],
                'title': response['snippet']['title'],
                'url': f"https://www.youtube.com/watch?v={response['id']}"
            }
            
        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred: {e.content.decode()}")
            return None
        except Exception as e:
            print(f"Upload failed: {e}")
            return None
    
    def update_video_privacy(self, video_id: str, privacy_status: str = "public") -> bool:
        """Update video privacy status"""
        
        if not self.youtube:
            if not self.authenticate():
                return False
        
        try:
            request = self.youtube.videos().update(
                part="status",
                body={
                    "id": video_id,
                    "status": {
                        "privacyStatus": privacy_status
                    }
                }
            )
            response = request.execute()
            print(f"Video {video_id} privacy updated to {privacy_status}")
            return True
            
        except HttpError as e:
            print(f"Failed to update privacy: {e}")
            return False
    
    def get_channel_info(self) -> Optional[Dict[str, Any]]:
        """Get channel information"""
        
        if not self.youtube:
            if not self.authenticate():
                return None
        
        try:
            request = self.youtube.channels().list(
                part="snippet,statistics",
                mine=True
            )
            response = request.execute()
            
            if response['items']:
                channel = response['items'][0]
                return {
                    'id': channel['id'],
                    'title': channel['snippet']['title'],
                    'description': channel['snippet']['description'],
                    'subscriber_count': channel['statistics'].get('subscriberCount', 0),
                    'video_count': channel['statistics'].get('videoCount', 0)
                }
            return None
            
        except HttpError as e:
            print(f"Failed to get channel info: {e}")
            return None 