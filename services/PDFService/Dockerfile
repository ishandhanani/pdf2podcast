FROM python:3.11-slim

WORKDIR /app

# Install only the necessary Python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    httpx \
    redis

# Copy shared package
COPY shared /shared
RUN pip install /shared

# Copy the service code
COPY services/PDFService/main.py ./

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]