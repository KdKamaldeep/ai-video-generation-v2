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
        
        # Define available story types
        self.story_types = {
            "motivation": "motivational with a touch of humor",
            "horror": "spooky and suspenseful",
            "horror_ghost": "spooky and suspenseful with a ghost",
            "horror_zombie": "spooky and suspenseful with a zombie",
            "horror_vampire": "spooky and suspenseful with a vampire",
            "horror_werewolf": "spooky and suspenseful with a werewolf",
            "horror_alien": "spooky and suspenseful with an alien",
            "horror_robot": "spooky and suspenseful with a robot",
            "horror_demon": "spooky and suspenseful with a demon",
            "horror_witch": "spooky and suspenseful with a witch",  
            "real_life": "relatable real-life situations",
            "adventure": "exciting and adventurous",
            "comedy": "funny and entertaining",
            "drama": "emotional and dramatic",
            "mystery": "mysterious and intriguing",
            "romance": "romantic and heartwarming",
            "sci_fi": "futuristic and sci-fi themed",
            "fantasy": "magical and fantastical"
        }
    
    def generate_script(self, story_type: str = "motivation") -> GeneratedScript:
        """Generate a YouTube Shorts script using OpenAI GPT with specified story type"""
        
        # Validate story type
        if story_type not in self.story_types:
            logger.warning(f"Invalid story type '{story_type}', using 'motivation' instead")
            story_type = "motivation"
        
        story_style = self.story_types[story_type]
        logger.info(f"Generating {story_type} script")
        
        prompt = self._create_prompt_for_story_type(story_type, story_style)
        
        try:
            logger.info(f"Sending request to OpenAI GPT-4 for {story_type} script")
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a creative script writer for YouTube Shorts. Create engaging, {story_style} content."},
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
            return self._generate_fallback_script(story_type)
    
    def _create_prompt_for_story_type(self, story_type: str, story_style: str) -> str:
        """Create a specific prompt for each story type"""
        
        base_prompt = f"""
                    Create a YouTube Shorts script for a channel called "Why Would You" that is themed around {story_type} content and uses a {story_style} storytelling style.

                    Your job is to write a short, emotionally engaging micro-story or scene that fits under 60 seconds. Follow these instructions:

                    1. Start with a compelling, curiosity-driven question that begins with "Why would you...?" — this is the hook that opens the story.
                    2. Build a short story or reflection around this question that feels real, cinematic, and emotionally resonant. Use visual language and small narrative moments.
                    3. Break the story into 3–5 short narration lines. Each line should include:
                       - "text": The narration sentence (keep it vivid and human-sounding)
                       - "duration": Duration in seconds (2–5s depending on pace and tension)
                       - "visual_suggestion": A visual idea that fits a vertical format — suggest camera angle, lighting, setting, mood (e.g., "over-the-shoulder shot of woman at dusk")

                    4. Keep content safe for all audiences, but allow emotional depth, mystery, tension, or inspiration depending on the story type.
                    5. Match tone and pacing to the {story_type} theme:
                       - Horror/thriller: Use short, tense lines and suspenseful reveals
                       - Motivational/inspirational: Use reflective lines with gradual emotional build
                       - Relatable/life lessons: Use authentic, grounded narration

                    6. Include:
                       - "title": A complete, catchy title that finishes the "Why Would You..." question
                       - "tags": 5 YouTube tags relevant to the story (shorts, emotional, horror, etc.)
                       - Optional: "cta": a final line that encourages reflection or audience response
                       - Optional: "twist": a surprise or ironic reveal at the end

                    Format your response strictly as a JSON object:

                    {{
                      "title": "Why Would You...?",
                      "narration": [
                        {{
                          "text": "Why would you...?",
                          "duration": 4.5,
                          "visual_suggestion": "..."
                        }},
                        ...
                      ],
                      "total_duration": 52.0,
                      "tags": ["shorts", "{story_type}", "emotional", "visual_storytelling", "whywouldyou"],
                      "twist": "...",        // optional
                      "cta": "..."           // optional
                    }}
                    """

        
        # Add specific instructions for each story type
        type_specific_instructions = {
            "motivation": "Focus on inspiring and uplifting content that motivates viewers to take action or improve themselves.",
            "horror": "Create suspenseful, spooky content that's scary but not too intense for YouTube. Use dark themes and mysterious elements.",
            "real_life": "Focus on everyday situations that people can relate to. Make it feel authentic and genuine.",
            "adventure": "Create exciting, action-packed content with thrilling scenarios and adventurous themes.",
            "comedy": "Focus on humor and entertainment. Make it funny and light-hearted with clever jokes.",
            "drama": "Create emotional, dramatic content that tugs at heartstrings and creates strong emotional responses.",
            "mystery": "Create intriguing, mysterious content that keeps viewers guessing and wanting to know more.",
            "romance": "Focus on love, relationships, and heartwarming romantic scenarios.",
            "sci_fi": "Create futuristic, sci-fi themed content with technology, space, and futuristic elements.",
            "fantasy": "Create magical, fantastical content with supernatural elements and magical themes."
        }
        
        specific_instruction = type_specific_instructions.get(story_type, "")
        if specific_instruction:
            base_prompt += f"\n\nAdditional {story_type} specific instruction: {specific_instruction}"
        
        return base_prompt
    
    def get_available_story_types(self) -> Dict[str, str]:
        """Get list of available story types"""
        return self.story_types.copy()
    
    def _generate_fallback_script(self, story_type: str) -> GeneratedScript:
        """Generate a fallback script if OpenAI API fails"""
        logger.info(f"Generating fallback script for {story_type}")
        
        # Fallback scripts for different story types
        fallback_scripts = {
            "motivation": {
                "title": "Why Would You Doubt Yourself?",
                "narration": [
                    ScriptLine(text="Why would you doubt yourself when you've already survived every bad day so far?", duration=4.5, visual_suggestion="person looking in mirror"),
                    ScriptLine(text="You're stronger than you think, and more capable than you believe.", duration=4.0, visual_suggestion="person overcoming obstacles"),
                    ScriptLine(text="So why would you doubt yourself? You've got this!", duration=3.5, visual_suggestion="person smiling confidently")
                ],
                "tags": ["motivation", "self-improvement", "confidence", "shorts", "inspiration"]
            },
            "horror": {
                "title": "Why Would You Go Into That Dark Room?",
                "narration": [
                    ScriptLine(text="Why would you go into that dark room when you know something's waiting?", duration=4.0, visual_suggestion="dark doorway with shadows"),
                    ScriptLine(text="Every horror movie ever made tells you not to do this.", duration=3.5, visual_suggestion="person hesitating at doorway"),
                    ScriptLine(text="But somehow, you still reach for that light switch...", duration=4.0, visual_suggestion="hand reaching for switch")
                ],
                "tags": ["horror", "spooky", "thriller", "shorts", "suspense"]
            },
            "real_life": {
                "title": "Why Would You Leave Your Phone on Silent?",
                "narration": [
                    ScriptLine(text="Why would you leave your phone on silent?", duration=3.0, visual_suggestion="person frantically searching for phone"),
                    ScriptLine(text="You know you're going to miss that important call.", duration=4.0, visual_suggestion="phone ringing with missed call notification"),
                    ScriptLine(text="But somehow, you still do it every single time.", duration=3.5, visual_suggestion="person facepalming in frustration")
                ],
                "tags": ["funny", "relatable", "phone", "shorts", "real_life"]
            },
            "adventure": {
                "title": "Why Would You Stay Home When Adventure Calls?",
                "narration": [
                    ScriptLine(text="Why would you stay home when adventure is calling your name?", duration=4.0, visual_suggestion="person looking out window at mountains"),
                    ScriptLine(text="The world is full of amazing places waiting to be discovered.", duration=4.5, visual_suggestion="travel montage"),
                    ScriptLine(text="So pack your bags and answer the call of adventure!", duration=3.5, visual_suggestion="person packing backpack")
                ],
                "tags": ["adventure", "travel", "exploration", "shorts", "wanderlust"]
            },
            "comedy": {
                "title": "Why Would You Try to Fix That Yourself?",
                "narration": [
                    ScriptLine(text="Why would you try to fix that yourself?", duration=3.0, visual_suggestion="person with broken item and tools"),
                    ScriptLine(text="You know you're going to make it worse.", duration=3.5, visual_suggestion="person making mess"),
                    ScriptLine(text="But here you are, with YouTube tutorials and hope.", duration=4.0, visual_suggestion="person watching tutorial on phone")
                ],
                "tags": ["comedy", "funny", "diy", "shorts", "relatable"]
            }
        }
        
        # Get fallback script for the story type, or use real_life as default
        fallback_data = fallback_scripts.get(story_type, fallback_scripts["real_life"])
        
        return GeneratedScript(
            title=fallback_data["title"],
            narration=fallback_data["narration"],
            total_duration=sum(line.duration for line in fallback_data["narration"]),
            tags=fallback_data["tags"]
        ) 