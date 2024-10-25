import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import io

app = FastAPI()

PDF_SERVICE_URL = "http://localhost:8000/convert"
AGENT_SERVICE_URL = "http://localhost:8964/transcribe"
TTS_SERVICE_URL = "http://localhost:8888/generate_tts"

class TranscriptionRequest(BaseModel):
    duration: int
    speaker_1_name: str
    speaker_2_name: str
    model: str

@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...), transcription_params: TranscriptionRequest = None):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Step 1: Convert PDF to Markdown
    pdf_response = requests.post(PDF_SERVICE_URL, files={"file": (file.filename, file.file, file.content_type)})
    if pdf_response.status_code != 200:
        raise HTTPException(status_code=500, detail="PDF conversion failed")
    markdown_content = pdf_response.text

    # Step 2: Process Markdown with Agent Service
    agent_payload = {
        "markdown": markdown_content,
        "duration": transcription_params.duration,
        "speaker_1_name": transcription_params.speaker_1_name,
        "speaker_2_name": transcription_params.speaker_2_name,
        "model": transcription_params.model
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