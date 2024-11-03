import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Response, BackgroundTasks
import json
import os
from typing import Dict, Optional
import time
from enum import Enum
from pydantic import BaseModel

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobStatusResponse(BaseModel):
    status: str
    message: str
    timestamp: float

class Job:
    def __init__(self):
        self.status = JobStatus.PENDING
        self.message = "Job started"
        self.timestamp = time.time()
        self.result: Optional[bytes] = None

app = FastAPI(debug=True)

PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://localhost:8003/convert")
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://localhost:8964/transcribe")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8888/generate_tts")

# In-memory job storage
jobs: Dict[str, Job] = {}

def update_job_status(job_id: str, status: JobStatus, message: str):
    if job_id not in jobs:
        raise KeyError(f"Job {job_id} not found")
    
    job = jobs[job_id]
    job.status = status
    job.message = message
    job.timestamp = time.time()

def process_pdf_task(job_id: str, file_content: bytes, transcription_params: dict):
    try:
        update_job_status(
            job_id, 
            JobStatus.PROCESSING, 
            "Converting PDF to markdown..."
        )
        
        pdf_response = requests.post(
            PDF_SERVICE_URL, 
            files={"file": ("file.pdf", file_content, "application/pdf")}
        )
        if pdf_response.status_code != 200:
            raise Exception("PDF conversion failed")
        markdown_content = pdf_response.text

        update_job_status(
            job_id, 
            JobStatus.PROCESSING, 
            "Processing with agent service..."
        )
        
        agent_payload = {
            "markdown": markdown_content,
            **transcription_params
        }
        agent_response = requests.post(AGENT_SERVICE_URL, json=agent_payload)
        if agent_response.status_code != 200:
            raise Exception("Agent processing failed")
        agent_result = agent_response.json()

        update_job_status(
            job_id, 
            JobStatus.PROCESSING, 
            "Generating text-to-speech..."
        )
        
        tts_payload = {
            "dialogue": agent_result["dialogue"]
        }
        tts_response = requests.post(TTS_SERVICE_URL, json=tts_payload)
        if tts_response.status_code != 200:
            raise Exception("TTS generation failed")

        # Store result and update status
        jobs[job_id].result = tts_response.content
        update_job_status(
            job_id, 
            JobStatus.COMPLETED, 
            "Processing completed"
        )

    except Exception as e:
        update_job_status(
            job_id, 
            JobStatus.FAILED, 
            str(e)
        )

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

    # Create job ID using timestamp
    job_id = str(int(time.time()))
    file_content = await file.read()
    
    # Initialize job
    jobs[job_id] = Job()

    # Start background process
    background_tasks.add_task(process_pdf_task, job_id, file_content, params)

    return {"job_id": job_id}

@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatusResponse(
        status=job.status.value,
        message=job.message,
        timestamp=job.timestamp
    )

@app.get("/output/{job_id}")
async def get_output(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=425,  # Too Early
            detail="Job is not completed yet"
        )
    
    if not job.result:
        raise HTTPException(
            status_code=500,
            detail="Job completed but no result found"
        )
    
    return Response(content=job.result, media_type="audio/mpeg")

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs[job_id]
    return {"message": f"Job {job_id} deleted"}

# Optional cleanup endpoint
@app.post("/cleanup")
async def cleanup_jobs():
    current_time = time.time()
    removed = 0
    for job_id in list(jobs.keys()):
        if current_time - jobs[job_id].timestamp > 3600:  # 1 hour
            del jobs[job_id]
            removed += 1
    return {"message": f"Removed {removed} old jobs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)