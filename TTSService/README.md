docker build -t tts-service .
docker run --gpus all -p 8888:8888 tts-service