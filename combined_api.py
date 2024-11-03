from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Response
from job_status import JobStatusManager, JobStatus
import requests
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
job_manager = JobStatusManager()

# Service URLs
PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://localhost:8003")
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://localhost:8964")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8888")

def process_pdf_task(job_id: str, file_content: bytes, transcription_params: dict):
    try:
        # Step 1: PDF Service
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Starting PDF conversion...")
        pdf_response = requests.post(
            f"{PDF_SERVICE_URL}/convert",
            files={"file": ("file.pdf", file_content, "application/pdf")},
            data={"job_id": job_id}
        )
        
        while True:
            status = requests.get(f"{PDF_SERVICE_URL}/status/{job_id}").json()
            job_manager.update_status(job_id, JobStatus.PROCESSING, f"PDF Service: {status['message']}")
            if status["status"] == "completed":
                markdown_content = requests.get(f"{PDF_SERVICE_URL}/output/{job_id}").text
                break
            elif status["status"] == "failed":
                raise Exception(f"PDF conversion failed: {status['message']}")
            time.sleep(1)

        # Step 2: Agent Service
        agent_payload = {
            "markdown": markdown_content,
            "job_id": job_id,
            **transcription_params
        }
        requests.post(f"{AGENT_SERVICE_URL}/transcribe", json=agent_payload)
        
        while True:
            status = requests.get(f"{AGENT_SERVICE_URL}/status/{job_id}").json()
            job_manager.update_status(job_id, JobStatus.PROCESSING, f"Agent Service: {status['message']}")
            if status["status"] == "completed":
                agent_result = requests.get(f"{AGENT_SERVICE_URL}/output/{job_id}").json()
                break
            elif status["status"] == "failed":
                raise Exception(f"Agent processing failed: {status['message']}")
            time.sleep(1)

        # Step 3: TTS Service
        tts_payload = {
            "dialogue": agent_result["dialogue"],
            "job_id": job_id
        }
        requests.post(f"{TTS_SERVICE_URL}/generate_tts", json=tts_payload)
        
        while True:
            status = requests.get(f"{TTS_SERVICE_URL}/status/{job_id}").json()
            job_manager.update_status(job_id, JobStatus.PROCESSING, f"TTS Service: {status['message']}")
            if status["status"] == "completed":
                result = requests.get(f"{TTS_SERVICE_URL}/output/{job_id}").content
                job_manager.set_result(job_id, result)
                job_manager.update_status(job_id, JobStatus.COMPLETED, "Processing completed successfully")
                break
            elif status["status"] == "failed":
                raise Exception(f"TTS generation failed: {status['message']}")
            time.sleep(1)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        job_manager.update_status(job_id, JobStatus.FAILED, str(e))

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
    job_manager.create_job(job_id)
    
    # Start processing
    file_content = await file.read()
    background_tasks.add_task(process_pdf_task, job_id, file_content, params)

    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    return job_manager.get_status(job_id)

@app.get("/output/{job_id}")
async def get_output(job_id: str):
    result = job_manager.get_result(job_id)
    return Response(content=result, media_type="audio/mpeg")

@app.post("/cleanup")
async def cleanup_jobs():
    removed = job_manager.cleanup_old_jobs()
    return {"message": f"Removed {removed} old jobs"}