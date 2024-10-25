# Build the Docker image
docker build -t pdf-conversion-service .

# Run the Docker container
docker run --gpus all -p 8000:8000 pdf-conversion-service