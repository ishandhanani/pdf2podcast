from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse
from docling.document_converter import DocumentConverter
import tempfile
import os
import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

def convert_pdf_to_markdown(pdf_path: str) -> str:
    logging.info(f"Converting PDF to Markdown: {pdf_path}")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()

@app.post("/convert", response_class=PlainTextResponse)
async def convert_pdf(file: UploadFile = File(...)):
    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        # Write the uploaded file content to the temporary file
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # Convert the PDF to markdown
        markdown_content = convert_pdf_to_markdown(temp_file_path)
        return markdown_content
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
