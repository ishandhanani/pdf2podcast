# Use NVIDIA PyTorch image as the base
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Install OpenGL libraries
RUN apt-get update \
    && apt-get install -y libgl1 libglib2.0-0 curl wget git procps \
    && apt-get clean

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download models
RUN python -c 'from deepsearch_glm.utils.load_pretrained_models import load_pretrained_nlp_models; load_pretrained_nlp_models(verbose=True);'

ADD sample.pdf /app/
ADD dry_run.py /app/
RUN python3 /app/dry_run.py

# Copy the Python script into the container
COPY main.py .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

