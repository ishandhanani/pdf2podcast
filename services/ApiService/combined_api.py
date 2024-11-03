from fastapi import HTTPException, FastAPI, File, UploadFile, Form, BackgroundTasks, Response
from shared.shared_types import ServiceType, JobStatus, StatusUpdate
import redis
import requests
import json
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(debug=True)
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"), decode_responses=True)

# Service URLs
PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://localhost:8003")
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://localhost:8964")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8888")

PDF_SERVICE_REFRESH = float(os.getenv("PDF_SERVICE_REFRESH", "0.1"))
AGENT_SERVICE_REFRESH = float(os.getenv("AGENT_SERVICE_REFRESH", "5"))
TTS_SERVICE_REFRESH = float(os.getenv("TTS_SERVICE_REFRESH", "0.1"))

def process_pdf_task(job_id: str, file_content: bytes, transcription_params: dict):
    try:
        pubsub = redis_client.pubsub()
        pubsub.subscribe("status_updates:all")

        # Start PDF Service
        requests.post(
            f"{PDF_SERVICE_URL}/convert",
            files={"file": ("file.pdf", file_content, "application/pdf")},
            data={"job_id": job_id}
        )
        
        # Monitor services
        current_service = ServiceType.PDF
        while True:
            message = pubsub.get_message()
            if message and message["type"] == "message":
                update = StatusUpdate.model_validate_json(message["data"])
                
                if update.job_id == job_id:
                    logger.info(f"Received update for job {job_id}: {update}")
                    
                    if update.status == JobStatus.FAILED:
                        raise Exception(f"{update.service}: {update.message}")
                    
                    if update.status == JobStatus.COMPLETED:
                        if current_service == ServiceType.PDF:
                            # Start Agent Service
                            markdown_content = requests.get(f"{PDF_SERVICE_URL}/output/{job_id}").text
                            requests.post(
                                f"{AGENT_SERVICE_URL}/transcribe", 
                                json={
                                    "markdown": markdown_content,
                                    "job_id": job_id,
                                    **transcription_params
                                }
                            )
                            current_service = ServiceType.AGENT
                            
                        elif current_service == ServiceType.AGENT:
                            # Start TTS Service
                            agent_result = requests.get(f"{AGENT_SERVICE_URL}/output/{job_id}").json()
                            requests.post(
                                f"{TTS_SERVICE_URL}/generate_tts", 
                                json={
                                    "dialogue": agent_result["dialogue"],
                                    "job_id": job_id
                                }
                            )
                            current_service = ServiceType.TTS
                            
                        elif current_service == ServiceType.TTS:
                            # Final output
                            return requests.get(f"{TTS_SERVICE_URL}/output/{job_id}").content

            time.sleep(0.01)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        raise

@app.post("/process_pdf", status_code=202)
async def process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    transcription_params: str = Form(...)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        params = json.loads(transcription_params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in transcription_params")

    # Create job
    job_id = str(int(time.time()))
    
    # Start processing
    file_content = await file.read()
    background_tasks.add_task(process_pdf_task, job_id, file_content, params)

    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get aggregated status from all services"""
    statuses = {}
    for service in ServiceType:
        status = redis_client.hgetall(f"status:{job_id}:{service}")
        if status:
            statuses[service] = status
    
    if not statuses:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return statuses

@app.get("/output/{job_id}")
async def get_output(job_id: str):
    """Get the final TTS output"""
    result = redis_client.get(f"result:{job_id}:{ServiceType.TTS}")
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return Response(content=result, media_type="audio/mpeg")

@app.post("/cleanup")
async def cleanup_jobs():
    """Clean up old jobs across all services"""
    removed = 0
    for service in ServiceType:
        pattern = f"status:*:{service}"
        for key in redis_client.scan_iter(match=pattern):
            job_id = key.split(":")[1]
            redis_client.delete(key)
            redis_client.delete(f"result:{job_id}:{service}")
            removed += 1
    return {"message": f"Removed {removed} old jobs"}