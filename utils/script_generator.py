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
        
        topic = "motivarion"

        prompt = """
                Create a YouTube Shorts script for a channel called "Why Would You" that is motivational with a touch of humor. Follow these instructions:

                1. Begin with an attention-grabbing question that starts with "Why would you...?" — make it motivational, not negative or sarcastic.
                2. Build a short story or reflection around this question that is inspiring, yet slightly funny or ironic.
                3. Break the narration into 3–5 short lines, each with:
                - "text": narration line
                - "duration": estimated time in seconds (total under 60)
                - "visual_suggestion": what viewers should see (stock footage, animation, etc.)
                4. Keep the content family-friendly, relatable, and emotionally resonant.
                5. Make sure the humor enhances the message, not distracts from it.
                6. Include a "title" completing the "Why Would You..." question in a catchy way.
                7. Add "tags": 5 YouTube tags (shorts, motivational, etc.)

                Format your response **strictly as JSON** like this:

                {
                "title": "Why Would You Doubt Yourself?",
                "narration": [
                    {
                    "text": "Why would you doubt yourself when you’ve already survived every bad day so far?",
                    "duration": 4.5,
                    "visual_suggestion": "person looking in mirror, cut to scenes of past struggles"
                    },
                    ...
                ],
                "total_duration": 52.0,
                "tags": ["shorts", "motivational", "funny", "self-improvement", "daily habits"]
                }
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