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

MAX_CONCURRENT_REQUESTS = 5
REQUEST_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

class TTSRequest(BaseModel):
    dialogue: List[Dict[str, str]]
    tts_api_key: str

class TTSService:
    def __init__(self):
        self.output_path = Path("sample.mp3")
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
    
    def _init_elevenlabs(self, tts_api_key: str) -> ElevenLabs:
        """Initialize ElevenLabs client with provided API key"""
        if not tts_api_key:
            raise ValueError("ElevenLabs API key is required")
        return ElevenLabs(api_key=tts_api_key)

    def _convert_text(self, text: str, voice_id: str, tts_api_key: str) -> bytes:
        """Convert text to speech using ElevenLabs with provided API key"""
        try:
            client = self._init_elevenlabs(tts_api_key)
            audio_stream = client.text_to_speech.convert(
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
            tasks = [
                (entry.get('text', ''), ELEVENLABS_VOICES[entry.get('speaker', 'speaker-1')], request.tts_api_key)
                for entry in request.dialogue
            ]
            
            for i in range(0, len(tasks), MAX_CONCURRENT_REQUESTS):
                batch = tasks[i:i + MAX_CONCURRENT_REQUESTS]
                futures = [
                    self.thread_pool.submit(self._convert_text, text, voice_id, api_key)
                    for text, voice_id, api_key in batch
                ]
                for future in futures:
                    combined_audio += future.result()

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
    _: None = Depends(get_request_semaphore)
):
    """Generate TTS audio from dialogue using ElevenLabs"""    
    try:
        output_path = await tts_service.process_parallel(request)
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
        "voices": list(ELEVENLABS_VOICES.keys()),
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS
    }