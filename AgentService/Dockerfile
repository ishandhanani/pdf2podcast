# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy service files
COPY main.py prompts.py job_status.py /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install fastapi uvicorn pydantic python-dotenv jinja2

# Install flexagent
COPY flexagent /app/flexagent
WORKDIR /app/flexagent/python
RUN pip install --use-pep517 -e .
WORKDIR /app

EXPOSE 8964

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8964"]