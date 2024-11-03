FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y libgl1 libglib2.0-0 curl wget git procps \
    && apt-get clean

WORKDIR /app

# Copy only requirements first to leverage caching
COPY services/PDFService/requirements.txt /app/
RUN pip install -r requirements.txt
RUN pip install redis

# Create the download script inline instead of copying
RUN echo 'from deepsearch_glm.utils.load_pretrained_models import load_pretrained_nlp_models\nload_pretrained_nlp_models(verbose=True)' > download_models.py
# Download models
RUN python download_models.py

# Copy remaining application files
COPY services/PDFService/main.py ./
COPY services/PDFService/sample.pdf ./
COPY services/PDFService/dry_run.py ./

# Copy shared package third
COPY shared /shared
RUN pip install /shared

# Run dry run
RUN python3 /app/dry_run.py

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
