FROM python:3.9-slim

WORKDIR /app

RUN pip install fastapi uvicorn python-multipart requests pydantic redis

# Copy shared package first
COPY shared /shared
RUN pip install /shared

# Copy service files
COPY services/APIService/main.py ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
