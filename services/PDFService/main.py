from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from shared.shared_types import ServiceType  # Import from shared_types
from shared.job_status import JobStatusManager, JobStatus
from fastapi.responses import PlainTextResponse
from docling.document_converter import DocumentConverter
from pydantic import BaseModel
import tempfile
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(debug=True)
job_manager = JobStatusManager(ServiceType.PDF)

class PDFRequest(BaseModel):
    job_id: str

def convert_pdf_to_markdown(pdf_path: str) -> str:
    logger.info(f"Converting PDF to Markdown: {pdf_path}")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()
    del converter
    del result
    return markdown

async def process_pdf(job_id: str, file_content: bytes):
    """Background task to process PDF conversion"""
    try:
        job_manager.update_status(
            job_id,
            JobStatus.PROCESSING,
            "Creating temporary file"
        )

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            job_manager.update_status(
                job_id,
                JobStatus.PROCESSING,
                "Converting PDF to Markdown"
            )
            
            # Convert the PDF to markdown
            markdown_content = convert_pdf_to_markdown(temp_file_path)
            
            # Store result
            job_manager.set_result(job_id, markdown_content.encode())
            job_manager.update_status(
                job_id,
                JobStatus.COMPLETED,
                "PDF conversion completed successfully"
            )

        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            logger.info(f"Cleaned up temporary file: {temp_file_path}")

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        job_manager.update_status(
            job_id,
            JobStatus.FAILED,
            f"PDF conversion failed: {str(e)}"
        )
        raise

@app.post("/convert", status_code=202)
async def convert_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_id: str = None
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Read file content
    content = await file.read()
    
    # Create job
    if not job_id:
        job_id = str(int(time.time()))
    
    job_manager.create_job(job_id)
    
    # Start processing in background
    background_tasks.add_task(process_pdf, job_id, content)
    
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get status of PDF conversion job"""
    return job_manager.get_status(job_id)

@app.get("/output/{job_id}", response_class=PlainTextResponse)
async def get_output(job_id: str):
    """Get the converted markdown content"""
    result = job_manager.get_result(job_id)
    return result.decode()

@app.get("/health")
async def health():
    try:
        # Try to initialize DocumentConverter
        converter = DocumentConverter()
        del converter
        
        return {
            "status": "healthy",
            "service": "pdf-converter"
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)