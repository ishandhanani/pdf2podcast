FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
        libgl1 \
        libglib2.0-0 \
        curl \
        wget \
        git \
        procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY services/PDFService/PDFModelService/requirements.txt /app/
RUN pip install -r requirements.txt \
    && pip install fastapi uvicorn python-multipart httpx

# Download required models
RUN echo 'from deepsearch_glm.utils.load_pretrained_models import load_pretrained_nlp_models\nload_pretrained_nlp_models(verbose=True)' > download_models.py
RUN python download_models.py

# Create the model API service
COPY services/PDFService/PDFModelService/main.py /app/

# Run validation script to ensure model loads correctly
RUN echo 'from docling.document_converter import DocumentConverter; DocumentConverter()' > validate.py
RUN python validate.py

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]