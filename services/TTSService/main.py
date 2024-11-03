from fastapi import FastAPI, BackgroundTasks, HTTPException
from shared.shared_types import ServiceType, JobStatus
from shared.job import JobStatusManager
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from elevenlabs.client import ElevenLabs
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DialogueEntry(BaseModel):
    text: str
    speaker: str
    voice_id: Optional[str] = None  # Optional voice_id override

class TTSRequest(BaseModel):
    dialogue: List[DialogueEntry]
    job_id: str
    voice_mapping: Dict[str, str]  # Maps speaker names to voice IDs

class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    description: Optional[str] = None

app = FastAPI(title="ElevenLabs TTS Service")
job_manager = JobStatusManager(ServiceType.TTS)
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))

@lru_cache(maxsize=1)
def get_available_voices() -> List[VoiceInfo]:
    """Fetch available voices from ElevenLabs API"""
    try:
        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        voices = client.voices.get_all()
        return [
            VoiceInfo(
                voice_id=voice.voice_id,
                name=voice.name,
                description=voice.description
            )
            for voice in voices
        ]
    except Exception as e:
        logger.error(f"Error fetching voices: {e}")
        return []

class TTSService:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)

    async def process_job(self, job_id: str, request: TTSRequest):
        try:
            # Validate voice mapping against available voices
            available_voices = {voice.voice_id for voice in get_available_voices()}
            invalid_voices = set(request.voice_mapping.values()) - available_voices
            
            if invalid_voices:
                raise ValueError(f"Invalid voice IDs: {invalid_voices}")

            job_manager.update_status(
                job_id, 
                JobStatus.PROCESSING,
                f"Processing {len(request.dialogue)} dialogue entries"
            )

            combined_audio = await self._process_dialogue(job_id, request.dialogue, request.voice_mapping)
            
            job_manager.set_result(job_id, combined_audio)
            job_manager.update_status(
                job_id,
                JobStatus.COMPLETED,
                "Audio generation completed successfully"
            )

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            job_manager.update_status(job_id, JobStatus.FAILED, str(e))

    async def _process_dialogue(
        self, 
        job_id: str, 
        dialogue: List[DialogueEntry], 
        voice_mapping: Dict[str, str]
    ) -> bytes:
        combined_audio = b""
        tasks = [
            (
                entry.text,
                entry.voice_id or voice_mapping[entry.speaker]  # Use override if provided
            )
            for entry in dialogue
        ]
        
        for i in range(0, len(tasks), MAX_CONCURRENT_REQUESTS):
            batch = tasks[i:i + MAX_CONCURRENT_REQUESTS]
            job_manager.update_status(
                job_id,
                JobStatus.PROCESSING,
                f"Processing batch {i//MAX_CONCURRENT_REQUESTS + 1} of {(len(tasks)-1)//MAX_CONCURRENT_REQUESTS + 1}"
            )
            
            futures = [
                self.thread_pool.submit(self._convert_text, text, voice_id)
                for text, voice_id in batch
            ]
            for future in futures:
                combined_audio += await asyncio.get_event_loop().run_in_executor(
                    None, future.result
                )
        
        return combined_audio

    def _convert_text(self, text: str, voice_id: str) -> bytes:
        """Convert text to speech using ElevenLabs"""
        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_monolingual_v1",
            output_format="mp3_44100_128",
            voice_settings={"stability": 0.5, "similarity_boost": 0.75, "style": 0.0}
        )
        return b"".join(chunk for chunk in audio_stream)

# Initialize service
tts_service = TTSService()

@app.get("/voices")
async def list_voices() -> List[VoiceInfo]:
    """Get list of available voices"""
    voices = get_available_voices()
    return voices

@app.post("/generate_tts", status_code=202)
async def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """Start TTS generation job"""
    job_manager.create_job(request.job_id)
    background_tasks.add_task(tts_service.process_job, request.job_id, request)
    return {"job_id": request.job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status"""
    return job_manager.get_status(job_id)

@app.get("/output/{job_id}")
async def get_output(job_id: str):
    """Get the generated audio file"""
    result = job_manager.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return Response(
        content=result,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=output.mp3"}
    )

@app.post("/cleanup")
async def cleanup_jobs():
    """Clean up old jobs"""
    removed = job_manager.cleanup_old_jobs()
    return {"message": f"Removed {removed} old jobs"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    voices = get_available_voices()
    return {
        "status": "healthy",
        "available_voices": len(voices),
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS
    }