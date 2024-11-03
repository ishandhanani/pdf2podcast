FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

RUN apt-get update \
    && apt-get install -y libgl1 libglib2.0-0 curl wget git procps \
    && apt-get clean

WORKDIR /app

# Copy only requirements first to leverage caching
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Create the download script inline instead of copying
RUN echo 'from deepsearch_glm.utils.load_pretrained_models import load_pretrained_nlp_models\nload_pretrained_nlp_models(verbose=True)' > download_models.py

# Download models
RUN python download_models.py

# Copy remaining application files after model download
COPY main.py job_status.py /app/
COPY sample.pdf /app/
COPY dry_run.py /app/
RUN python3 /app/dry_run.py

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]