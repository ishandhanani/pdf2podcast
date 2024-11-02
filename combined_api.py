import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Response, BackgroundTasks
import json
import os
from typing import Dict
import time

app = FastAPI(debug=True)

PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://localhost:8003/convert")
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://localhost:8964/transcribe")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8888/generate_tts")

# In-memory job storage
jobs: Dict[str, dict] = {}

def update_job_status(job_id: str, status: str, message: str):
    jobs[job_id].update({
        "status": status,
        "message": message,
        "timestamp": time.time()
    })

async def process_pdf_task(job_id: str, file_content: bytes, transcription_params: dict):
    try:
        update_job_status(job_id, "processing", "Converting PDF to markdown...")
        
        pdf_response = requests.post(
            PDF_SERVICE_URL, 
            files={"file": ("file.pdf", file_content, "application/pdf")}
        )
        if pdf_response.status_code != 200:
            raise Exception("PDF conversion failed")
        markdown_content = pdf_response.text

        update_job_status(job_id, "processing", "Processing with agent service...")
        
        agent_payload = {
            "markdown": markdown_content,
            **transcription_params
        }
        agent_response = requests.post(AGENT_SERVICE_URL, json=agent_payload)
        if agent_response.status_code != 200:
            raise Exception("Agent processing failed")
        agent_result = agent_response.json()

        update_job_status(job_id, "processing", "Generating text-to-speech...")
        
        tts_payload = {
            "dialogue": agent_result["dialogue"]
        }
        tts_response = requests.post(TTS_SERVICE_URL, json=tts_payload)
        if tts_response.status_code != 200:
            raise Exception("TTS generation failed")

        # Store result and update status
        jobs[job_id]["result"] = tts_response.content
        update_job_status(job_id, "completed", "Processing completed")

    except Exception as e:
        update_job_status(job_id, "failed", str(e))

@app.post("/process_pdf")
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

    # timestamp jobID
    job_id = str(int(time.time()))
    file_content = await file.read()
    
    jobs[job_id] = {
        "status": "pending",
        "message": "Job started",
        "timestamp": time.time(),
        "result": None
    }

    # Start background process
    background_tasks.add_task(process_pdf_task, job_id, file_content, params)

    return {"job_id": job_id, "status": "pending"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] == "completed" and job["result"]:
        return Response(content=job["result"], media_type="audio/mpeg")
    
    return {
        "status": job["status"],
        "message": job["message"]
    }

# # Simple cleanup - remove jobs older than 1 hour
# @app.get("/cleanup")
# async def cleanup_jobs():
#     current_time = time.time()
#     removed = 0
#     for job_id in list(jobs.keys()):
#         if current_time - jobs[job_id]["timestamp"] > 3600:
#             del jobs[job_id]
#             removed += 1
#     return {"message": f"Removed {removed} old jobs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)