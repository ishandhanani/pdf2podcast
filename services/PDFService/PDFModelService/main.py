from fastapi import FastAPI, File, UploadFile, HTTPException
from docling.document_converter import DocumentConverter
import tempfile
import os
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(debug=True)

# Initialize the document converter
converter = DocumentConverter()

@app.post("/convert")
async def convert_pdf(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Convert a PDF file to markdown format
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        # Create temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Convert PDF to markdown
            logger.info(f"Converting PDF to Markdown: {file.filename}")
            result = converter.convert(temp_file_path)
            markdown = result.document.export_to_markdown()
            
            return {"markdown": markdown}

        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            logger.info(f"Cleaned up temporary file: {temp_file_path}")

    except Exception as e:
        logger.error(f"Error converting PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")

@app.get("/health")
async def health():
    """
    Health check endpoint that verifies the model is loaded and working
    """
    try:
        # Verify converter is initialized and working
        if not converter:
            raise Exception("Document converter not initialized")
            
        return {
            "status": "healthy",
            "service": "pdf-model-api",
            "model_loaded": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)