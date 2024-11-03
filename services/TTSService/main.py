from fastapi import FastAPI, BackgroundTasks
from shared.shared_types import ServiceType, JobStatus
from shared.job import JobStatusManager
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict
import logging
from elevenlabs.client import ElevenLabs
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    dialogue: List[Dict[str, str]]
    job_id: str


app = FastAPI(title="ElevenLabs TTS Service")
job_manager = JobStatusManager(ServiceType.TTS)

# Constants
ELEVENLABS_VOICES = {
    "speaker-1": "iP95p4xoKVk53GoZ742B",
    "speaker-2": "9BWtsMINqrJLrRacOk9x",
}
MAX_CONCURRENT_REQUESTS = 5

class TTSService:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)

    async def process_job(self, job_id: str, request: TTSRequest):
        try:
            job_manager.update_status(
                job_id, 
                JobStatus.PROCESSING,
                f"Processing {len(request.dialogue)} dialogue entries"
            )

            combined_audio = await self._process_dialogue(job_id, request.dialogue)
            
            job_manager.set_result(job_id, combined_audio)
            job_manager.update_status(
                job_id,
                JobStatus.COMPLETED,
                "Audio generation completed successfully"
            )

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            job_manager.update_status(job_id, JobStatus.FAILED, str(e))

    async def _process_dialogue(self, job_id: str, dialogue: List[Dict[str, str]]) -> bytes:
        combined_audio = b""
        tasks = [
            (entry.get('text', ''), ELEVENLABS_VOICES[entry.get('speaker', 'speaker-1')])
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
    return {
        "status": "healthy",
        "voices": list(ELEVENLABS_VOICES.keys()),
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS
    }