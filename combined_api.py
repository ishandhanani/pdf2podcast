import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Response
from pydantic import BaseModel
import io
import json
import os

app = FastAPI(debug=True)

PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://localhost:8000/convert")
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://localhost:8964/transcribe")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8888/generate_tts")

class TranscriptionRequest(BaseModel):
    duration: int
    speaker_1_name: str
    speaker_2_name: str
    model: str
    api_key: str

@app.get("/health")
async def health():
    try:
        # Check all dependent services
        health_status = {
            "status": "healthy",
            "services": {
                "pdf": False,
                "agent": False,
                "tts": False
            }
        }
        
        # Check PDF service
        pdf_response = requests.get(f"{PDF_SERVICE_URL}/health", timeout=5)
        health_status["services"]["pdf"] = pdf_response.status_code == 200
        
        # Check Agent service  
        agent_response = requests.get(f"{AGENT_SERVICE_URL}/health", timeout=5)
        health_status["services"]["agent"] = agent_response.status_code == 200
        
        # Check TTS service
        tts_response = requests.get(f"{TTS_SERVICE_URL}/health", timeout=5)
        health_status["services"]["tts"] = tts_response.status_code == 200
        
        # Overall status is healthy only if all services are healthy
        if not all(health_status["services"].values()):
            health_status["status"] = "degraded"
            
        return health_status
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "services": health_status["services"]
        }

@app.post("/process_pdf")
async def process_pdf(
    file: UploadFile = File(...),
    transcription_params: str = Form(...)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Parse transcription_params from JSON string
    try:
        params = json.loads(transcription_params)
        transcription_request = TranscriptionRequest(**params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in transcription_params")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid transcription parameters")

    # Step 1: Convert PDF to Markdown
    pdf_response = requests.post(PDF_SERVICE_URL, files={"file": (file.filename, file.file, file.content_type)})
    if pdf_response.status_code != 200:
        raise HTTPException(status_code=500, detail="PDF conversion failed")
    markdown_content = pdf_response.text

    # Step 2: Process Markdown with Agent Service
    agent_payload = {
        "markdown": markdown_content,
        "duration": transcription_request.duration,
        "speaker_1_name": transcription_request.speaker_1_name,
        "speaker_2_name": transcription_request.speaker_2_name,
        "model": transcription_request.model,
        "api_key": transcription_request.api_key  # Pass through the API key
    }
    agent_response = requests.post(AGENT_SERVICE_URL, json=agent_payload)
    if agent_response.status_code != 200:
        raise HTTPException(status_code=500, detail="Agent processing failed")
    agent_result = agent_response.json()

    # Step 3: Generate TTS
    tts_response = requests.post(TTS_SERVICE_URL, json=agent_result)
    if tts_response.status_code != 200:
        raise HTTPException(status_code=500, detail="TTS generation failed")

    # Return the audio file
    return Response(content=tts_response.content, media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)