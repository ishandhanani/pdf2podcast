import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from elevenlabs.client import ElevenLabs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ELEVENLABS_VOICES = {
    "speaker-1": "iP95p4xoKVk53GoZ742B",  # Chris - casual conversational
    "speaker-2": "9BWtsMINqrJLrRacOk9x",  # Aria - expressive social media
}

# Configure rate limiting
MAX_CONCURRENT_REQUESTS = 5  # Maximum number of concurrent requests
REQUEST_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

class TTSRequest(BaseModel):
    dialogue: List[Dict[str, str]]

class TTSService:
    def __init__(self):
        self.output_path = Path("sample.mp3")
        self.elevenlabs_client = self._init_elevenlabs()
        # Limit concurrent threads in the thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
    
    def _init_elevenlabs(self) -> ElevenLabs:
        """Initialize ElevenLabs client"""
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is not set")
        return ElevenLabs(api_key=api_key)

    def _convert_text(self, text: str, voice_id: str) -> bytes:
        """Convert text to speech using ElevenLabs"""
        try:
            audio_stream = self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_monolingual_v1",
                output_format="mp3_44100_128",
                voice_settings={
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                }
            )
            return b"".join(chunk for chunk in audio_stream)
        except Exception as e:
            logger.error(f"ElevenLabs API error: {str(e)}")
            raise

    async def process_parallel(self, request: TTSRequest) -> Path:
        """Process TTS request using ElevenLabs in parallel with limited concurrency"""
        logger.info(f"Processing {len(request.dialogue)} dialogues with ElevenLabs (parallel)")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "output.mp3"
            
            combined_audio = b""
            # Convert to list of tuples for easier processing
            tasks = [
                (entry.get('text', ''), ELEVENLABS_VOICES[entry.get('speaker', 'speaker-1')])
                for entry in request.dialogue
            ]
            
            # Process in batches to maintain rate limits
            for i in range(0, len(tasks), MAX_CONCURRENT_REQUESTS):
                batch = tasks[i:i + MAX_CONCURRENT_REQUESTS]
                futures = [
                    self.thread_pool.submit(self._convert_text, text, voice_id)
                    for text, voice_id in batch
                ]
                for future in futures:
                    combined_audio += future.result()

            temp_path.write_bytes(combined_audio)
            self.output_path.write_bytes(temp_path.read_bytes())

        return self.output_path

    async def process_sequential(self, request: TTSRequest) -> Path:
        """Process TTS request using ElevenLabs sequentially"""
        logger.info(f"Processing {len(request.dialogue)} dialogues with ElevenLabs (sequential)")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "output.mp3"

            combined_audio = b""
            for entry in request.dialogue:
                text = entry.get('text', '')
                speaker = entry.get('speaker', 'speaker-1')
                voice_id = ELEVENLABS_VOICES[speaker]

                audio_bytes = self._convert_text(text, voice_id)
                combined_audio += audio_bytes

            temp_path.write_bytes(combined_audio)
            self.output_path.write_bytes(temp_path.read_bytes())

        return self.output_path

app = FastAPI(title="ElevenLabs TTS Service", debug=True)
tts_service = TTSService()

async def get_request_semaphore():
    """Dependency to manage concurrent requests"""
    async with REQUEST_SEMAPHORE:
        yield

@app.post("/generate_tts")
async def generate_tts(
    request: TTSRequest,
    _: None = Depends(get_request_semaphore)  # Add rate limiting dependency
):
    """
    Generate TTS audio from dialogue using ElevenLabs
    
    Rate-limited to MAX_CONCURRENT_REQUESTS concurrent requests
    """
    PARALLEL_PROCESSING = True
    
    try:
        if PARALLEL_PROCESSING:
            output_path = await tts_service.process_parallel(request)
        else:
            output_path = await tts_service.process_sequential(request)

        return FileResponse(
            output_path,
            media_type="audio/mpeg",
            filename="output.mp3"
        )
    except Exception as e:
        logger.error(f"Error processing TTS request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_tts/health")
async def health():
    return {
        "status": "healthy",
        "voices": list(ELEVENLABS_VOICES.keys()),  # List available voices
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS  # Show configuration
    }