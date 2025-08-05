import os
import json
import logging
from typing import List, Dict, Any
import openai
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ScriptLine(BaseModel):
    text: str
    duration: float
    visual_suggestion: str

class GeneratedScript(BaseModel):
    title: str
    narration: List[ScriptLine]
    total_duration: float
    tags: List[str]

class ScriptGenerator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        logger.info("Initializing OpenAI client")
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_script(self) -> GeneratedScript:
        """Generate a YouTube Shorts script using OpenAI GPT"""
        
        prompt = """
        Create a YouTube Shorts script for a channel called "Why Would You" that follows these requirements:
        
        1. Start with a question like "Why Would You...?" that's intriguing and relatable
        2. Create a short story that's engaging and under 60 seconds when narrated
        3. Include 3-5 narration lines with timing suggestions
        4. Each line should have a visual suggestion for stock images
        5. Make it humorous, relatable, or thought-provoking
        6. Include relevant tags for YouTube
        
        Format the response as JSON with:
        {
            "title": "Why Would You... [engaging title]",
            "narration": [
                {
                    "text": "narration text",
                    "duration": 3.5,
                    "visual_suggestion": "person looking confused at computer"
                }
            ],
            "total_duration": 45.0,
            "tags": ["funny", "relatable", "shorts"]
        }
        
        Make sure the total duration is under 60 seconds and the content is family-friendly.
        """
        
        try:
            logger.info("Sending request to OpenAI GPT-4")
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a creative script writer for YouTube Shorts. Create engaging, relatable content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content
            logger.info("Received response from OpenAI, parsing JSON")
            script_data = json.loads(content)
            
            # Convert to Pydantic model
            narration_lines = [ScriptLine(**line) for line in script_data["narration"]]
            logger.info(f"Script parsed successfully: {script_data['title']}")
            
            return GeneratedScript(
                title=script_data["title"],
                narration=narration_lines,
                total_duration=script_data["total_duration"],
                tags=script_data["tags"]
            )
            
        except Exception as e:
            logger.error(f"Failed to generate script with OpenAI: {str(e)}")
            logger.info("Using fallback script")
            # Fallback script if API fails
            return self._generate_fallback_script()
    
    def _generate_fallback_script(self) -> GeneratedScript:
        """Generate a fallback script if OpenAI API fails"""
        logger.info("Generating fallback script")
        return GeneratedScript(
            title="Why Would You Leave Your Phone on Silent?",
            narration=[
                ScriptLine(
                    text="Why would you leave your phone on silent?",
                    duration=3.0,
                    visual_suggestion="person frantically searching for phone"
                ),
                ScriptLine(
                    text="You know you're going to miss that important call.",
                    duration=4.0,
                    visual_suggestion="phone ringing with missed call notification"
                ),
                ScriptLine(
                    text="But somehow, you still do it every single time.",
                    duration=3.5,
                    visual_suggestion="person facepalming in frustration"
                )
            ],
            total_duration=10.5,
            tags=["funny", "relatable", "phone", "shorts"]
        ) 