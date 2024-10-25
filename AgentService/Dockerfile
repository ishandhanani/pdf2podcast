# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic python-dotenv jinja2

# Install flexagent
COPY flexagent /app/flexagent
WORKDIR /app/flexagent/python
RUN pip install --use-pep517 -e .
WORKDIR /app

# Make port 8964 available to the world outside this container
EXPOSE 8964

# Run the API when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8964"]

