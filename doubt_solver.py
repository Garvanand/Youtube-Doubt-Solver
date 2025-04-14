import os
from typing import Optional, List, Dict
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
from dotenv import load_dotenv
import requests
import json
import re
import time

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

class DoubtSolver:
    def __init__(self):
        self.video_content = None
        self.transcript = None
        self.available_languages = []
        self.current_language = 'en'  
        
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL with improved pattern matching."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com\/e\/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com\/user\/[^\/]+\/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com\/[^\/]+\/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com\/watch\?feature=player_embedded&v=)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com\/\?v=)([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def validate_url(self, url: str) -> bool:
        """Validate if the URL is accessible and contains a valid video."""
        try:
            response = requests.head(url, allow_redirects=True)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_available_languages(self, video_id: str) -> List[str]:
        """Get list of available transcript languages."""
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            languages = [transcript.language_code for transcript in transcript_list]
            return languages
        except Exception:
            return []

    def get_transcript(self, video_id: str, language: str = 'en') -> Optional[str]:
        """Get transcript in specified language with fallbacks."""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
            return " ".join([item['text'] for item in transcript_list])
        except (TranscriptsDisabled, NoTranscriptFound):
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                return " ".join([item['text'] for item in transcript_list])
            except Exception:
                return None

    def get_video_content(self, url: str, max_retries: int = 3) -> bool:
        """Get video content and transcript with retry mechanism."""
        if not self.validate_url(url):
            print("Invalid or inaccessible YouTube URL")
            return False

        video_id = self.extract_video_id(url)
        if not video_id:
            print("Could not extract video ID from URL")
            return False

        for attempt in range(max_retries):
            try:
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': True,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    if not info:
                        raise Exception("Could not extract video information")

                    self.video_content = {
                        'title': info.get('title', 'Unknown Title'),
                        'description': info.get('description', 'No description available'),
                        'author': info.get('uploader', 'Unknown Author'),
                        'duration': info.get('duration', 0),
                        'view_count': info.get('view_count', 0),
                        'upload_date': info.get('upload_date', 'Unknown')
                    }

                self.available_languages = self.get_available_languages(video_id)
                self.transcript = self.get_transcript(video_id, self.current_language)
                
                return True

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error processing video after {max_retries} attempts: {str(e)}")
                    return False
                print(f"Attempt {attempt + 1} failed. Retrying...")
                time.sleep(2 ** attempt)  

        return False

    def generate_content(self, prompt: str) -> str:
        """Generate content using Gemini API."""
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            response = requests.post(
                GEMINI_API_URL,
                headers=headers,
                data=json.dumps(data)
            )
            response.raise_for_status()
            
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                if 'content' in result['candidates'][0]:
                    parts = result['candidates'][0]['content'].get('parts', [])
                    if parts and 'text' in parts[0]:
                        return parts[0]['text']
            
            return "Sorry, I couldn't generate a response. Please try again."
            
        except requests.exceptions.RequestException as e:
            return f"Error calling Gemini API: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def answer_question(self, question: str) -> str:
        """Answer question based on video content and transcript."""
        if not self.video_content:
            return "Please provide a YouTube URL first."

        context = f"""
        Video Title: {self.video_content['title']}
        Video Description: {self.video_content['description']}
        Author: {self.video_content['author']}
        Duration: {self.video_content['duration']} seconds
        Upload Date: {self.video_content['upload_date']}
        """

        if self.transcript:
            context += f"\nVideo Transcript: {self.transcript}"

        prompt = f"""
        You are a helpful assistant that answers questions about a specific YouTube video.
        You can answer questions based on both the video's metadata (title, description) and its transcript.
        If the question is not related to the video content at all, respond with "Sorry, I can only answer questions about this video."

        {context}

        Question: {question}

        Please provide a clear and concise answer based on the available information.
        If the answer comes from the transcript, mention that. If it comes from the video description or title, mention that as well.
        """

        return self.generate_content(prompt)

def main():
    try:
        solver = DoubtSolver()
    except Exception as e:
        print(f"Failed to initialize the doubt solver: {str(e)}")
        return
    
    print("Welcome to YouTube Doubt Solver!")
    print("Please enter a YouTube URL:")
    url = input().strip()
    
    if not solver.get_video_content(url):
        return
    
    print("\nVideo processed successfully!")
    print(f"Title: {solver.video_content['title']}")
    
    if solver.available_languages:
        print("\nAvailable transcript languages:", ", ".join(solver.available_languages))
        print("Current language:", solver.current_language)
    else:
        print("\nNo transcript available, but you can still ask questions about the video's title and description.")
    
    print("\nYou can now ask questions about the video. Type 'exit' to quit.")
    
    while True:
        print("\nYour question:")
        question = input().strip()
        
        if question.lower() == 'exit':
            break
            
        answer = solver.answer_question(question)
        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    main() 