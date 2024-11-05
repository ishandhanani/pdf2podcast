from fastapi import HTTPException, FastAPI, File, UploadFile, Form, BackgroundTasks, Response, WebSocket, WebSocketDisconnect
from shared.shared_types import ServiceType, JobStatus, StatusUpdate
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Set
from minio import Minio
from minio.error import S3Error
import redis
import requests
import json
import os
import logging
import time
import asyncio
from collections import defaultdict
from threading import Thread
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionParams(BaseModel):
    duration: int = Field(..., description="Duration in minutes")
    speaker_1_name: str = Field(..., description="Name of the first speaker")
    speaker_2_name: str = Field(..., description="Name of the second speaker")
    model: str = Field(..., description="Model name/path to use for transcription")
    voice_mapping: Dict[str, str] = Field(
        ..., 
        description="Mapping of speaker IDs to voice IDs",
        example={
            "speaker-1": "iP95p4xoKVk53GoZ742B",
            "speaker-2": "9BWtsMINqrJLrRacOk9x"
        }
    )

app = FastAPI(debug=True)
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"), decode_responses=False)

# Service URLs
PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://localhost:8003")
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://localhost:8964")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8889")

# Minio config
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "audio-results")

# MP3 Cache TTL
MP3_CACHE_TTL = 60 * 60 * 4  # 4 hours

# CORS setup
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in CORS_ORIGINS.split(",")]
logger.info(f"Configuring CORS with allowed origins: {allowed_origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=3600,
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.pubsub = None
        self.message_queue = queue.Queue()
        self.redis_thread = None
        self.should_stop = False
        
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        self.active_connections[job_id].add(websocket)
        logger.info(f"New WebSocket connection for job {job_id}. Total connections: {len(self.active_connections[job_id])}")
        
        # Start Redis listener if not already running
        if self.redis_thread is None:
            self.redis_thread = Thread(target=self._redis_listener)
            self.redis_thread.daemon = True
            self.redis_thread.start()
            # Start the async message processor
            asyncio.create_task(self._process_messages())
        
    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
            logger.info(f"WebSocket disconnected for job {job_id}. Remaining connections: {len(self.active_connections[job_id]) if job_id in self.active_connections else 0}")

    def _redis_listener(self):
        """Redis subscription running in a separate thread"""
        try:
            self.pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
            self.pubsub.subscribe("status_updates:all")
            logger.info("Successfully subscribed to Redis status_updates:all channel")
            
            while not self.should_stop:
                message = self.pubsub.get_message()
                if message and message['type'] == 'message':
                    self.message_queue.put(message['data'])
                time.sleep(0.01)  # Prevent tight loop
                
        except Exception as e:
            logger.error(f"Redis subscription error: {e}")
        finally:
            if self.pubsub:
                self.pubsub.unsubscribe()
                self.pubsub.close()

    async def _process_messages(self):
        """Async task to process messages from the queue and broadcast them"""
        while True:
            try:
                # Check queue in a non-blocking way
                while not self.message_queue.empty():
                    message = self.message_queue.get_nowait()
                    try:
                        if isinstance(message, bytes):
                            message = message.decode('utf-8')
                        
                        update = json.loads(message)
                        job_id = update.get('job_id')
                        
                        if job_id and job_id in self.active_connections:
                            await self.broadcast_to_job(
                                job_id,
                                {
                                    'service': update.get('service'),
                                    'status': update.get('status'),
                                    'message': update.get('message', '')
                                }
                            )
                            logger.info(f"Broadcasted update for job {job_id}: {update.get('service')} - {update.get('status')}")
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in Redis message: {message}")
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
                        
                # Small delay before next check
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await asyncio.sleep(1)

    async def broadcast_to_job(self, job_id: str, message: dict):
        """Send message to all WebSocket connections for a job"""
        if job_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except WebSocketDisconnect:
                    disconnected.add(connection)
                except Exception as e:
                    logger.error(f"Error sending message to WebSocket: {e}")
                    disconnected.add(connection)
            
            # Clean up disconnected clients
            for connection in disconnected:
                self.disconnect(connection, job_id)

    def cleanup(self):
        """Cleanup resources"""
        self.should_stop = True
        if self.redis_thread:
            self.redis_thread.join(timeout=1.0)
        if self.pubsub:
            self.pubsub.close()

@app.websocket("/ws/status/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    try:
        # Accept the WebSocket connection
        await manager.connect(websocket, job_id)
        
        # Send initial status for all services
        for service in ServiceType:
            status_data = redis_client.hgetall(f"status:{job_id}:{service}")
            if status_data:
                status_msg = {
                    "service": service.value,
                    "status": status_data.get(b"status", b"").decode(),
                    "message": status_data.get(b"message", b"").decode()
                }
                await websocket.send_json(status_msg)
                logger.info(f"Sent initial status for {job_id} {service}: {status_msg}")
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (ping/pong handled automatically by FastAPI)
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
            
            await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        manager.disconnect(websocket, job_id)

# Initialize the connection manager
manager = ConnectionManager()

# TODO: use this to wrap redis as well
# TODO: wrap errors in StorageError
# TODO: implement cleanup and delete as well
class StorageManager:
    def __init__(self):
        """Initialize MinIO client and ensure bucket exists"""
        try:
            self.client = Minio(
                os.getenv("MINIO_ENDPOINT", "minio:9000"),
                access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
                secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
                secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
            )
            
            self.bucket_name = os.getenv("MINIO_BUCKET_NAME", "audio-results")
            self._ensure_bucket_exists()
            logger.info("Successfully initialized MinIO storage")
            
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            raise

    def _ensure_bucket_exists(self):
        try:    
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except Exception as e:
            logger.error(f"Failed to ensure bucket exists: {e}")
            raise
    
    def store_audio(self, job_id: str, audio_content: bytes, filename: str, transcription_params: TranscriptionParams):
        try:
            object_name = f"{job_id}/{filename}"
            self.client.put_object(self.bucket_name, object_name, io.BytesIO(audio_content), len(audio_content), content_type="audio/mpeg")
            logger.info(f"Stored audio for {job_id} in MinIO as {object_name}")
        except S3Error as e:
            logger.error(f"Failed to store audio in MinIO: {e}")
            raise
            
    def get_audio(self, job_id: str, filename: str):
        try:
            object_name = f"{job_id}/{filename}"
            result = self.client.get_object(self.bucket_name, object_name).read()
            logger.info(f"Retrieved audio for {job_id} from MinIO as {object_name}")
            return result
        except S3Error as e:
            logger.error(f"Failed to get audio from MinIO: {e}")
            raise

storage_manager = StorageManager()  

def process_pdf_task(job_id: str, file_content: bytes, transcription_params: TranscriptionParams):
    try:
        pubsub = redis_client.pubsub()
        pubsub.subscribe("status_updates:all")

        # Start PDF Service
        requests.post(
            f"{PDF_SERVICE_URL}/convert",
            files={"file": ("file.pdf", file_content, "application/pdf")},
            data={"job_id": job_id}
        )

        # Monitor services
        current_service = ServiceType.PDF
        while True:
            message = pubsub.get_message()
            if message and message["type"] == "message":
                update = StatusUpdate.model_validate_json(message["data"].decode())
                
                if update.job_id == job_id:
                    logger.info(f"Received update for job {job_id}: {update}")
                    
                    if update.status == JobStatus.FAILED:
                        raise Exception(f"{update.service}: {update.message}")
                    
                    if update.status == JobStatus.COMPLETED:
                        if current_service == ServiceType.PDF:
                            # Start Agent Service
                            markdown_content = requests.get(f"{PDF_SERVICE_URL}/output/{job_id}").text
                            requests.post(
                                f"{AGENT_SERVICE_URL}/transcribe", 
                                json={
                                    "markdown": markdown_content,
                                    "job_id": job_id,
                                    **transcription_params.model_dump()
                                }
                            )
                            current_service = ServiceType.AGENT
                            
                        elif current_service == ServiceType.AGENT:
                            # Start TTS Service
                            agent_result = requests.get(f"{AGENT_SERVICE_URL}/output/{job_id}").json()
                            requests.post(
                                f"{TTS_SERVICE_URL}/generate_tts", 
                                json={
                                    "dialogue": agent_result["dialogue"],
                                    "job_id": job_id,
                                    "voice_mapping": transcription_params.voice_mapping  # Forward the voice mapping
                                }
                            )
                            current_service = ServiceType.TTS
                            
                        elif current_service == ServiceType.TTS:
                            # Get final output and store it
                            logger.info(f"TTS completed for {job_id}, fetching and storing result")
                            audio_content = requests.get(f"{TTS_SERVICE_URL}/output/{job_id}").content
                            
                            # Store both the content and the ready flag
                            redis_client.set(f"result:{job_id}:{ServiceType.TTS}", audio_content, ex=MP3_CACHE_TTL)
                            redis_client.set(f"final_status:{job_id}", "ready", ex=MP3_CACHE_TTL)
                            
                            # Store in DB
                            storage_manager.store_audio(job_id, audio_content, f"{job_id}.mp3", transcription_params)

                            logger.info(f"Stored TTS result for {job_id}, size: {len(audio_content)} bytes, with TTL: {MP3_CACHE_TTL} seconds")
                            return audio_content

            time.sleep(0.01)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        raise

@app.post("/process_pdf", status_code=202)
async def process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    transcription_params: str = Form(...)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        params_dict = json.loads(transcription_params)
        params = TranscriptionParams.model_validate(params_dict)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in transcription_params")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create job
    job_id = str(int(time.time()))
    
    # Start processing
    file_content = await file.read()
    background_tasks.add_task(process_pdf_task, job_id, file_content, params)

    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get aggregated status from all services"""
    statuses = {}
    for service in ServiceType:
        status = redis_client.hgetall(f"status:{job_id}:{service}")
        if status:
            # Decode the bytes to strings
            statuses[service] = {k.decode(): v.decode() for k, v in status.items()}
    
    if not statuses:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return statuses

# This needs to also interact with our db as well. Check cache first if job running. If nothing there, check db  
@app.get("/output/{job_id}")
async def get_output(job_id: str):
    """Get the final TTS output"""
    # First check if the final result is ready
    is_ready = redis_client.get(f"final_status:{job_id}")
    if not is_ready:
        # Check if TTS service reports completion
        tts_status = redis_client.hgetall(f"status:{job_id}:{ServiceType.TTS}")
        if not tts_status or tts_status.get(b"status", b"").decode() != "completed":
            raise HTTPException(status_code=404, detail="Result not found")
            
        # If TTS reports complete but result not ready, it's still being fetched
        raise HTTPException(
            status_code=425,  # Too Early
            detail="Result is being prepared"
        )

    result = redis_client.get(f"result:{job_id}:{ServiceType.TTS}")
    if not result:
        logger.info(f"Final result not found in cache for {job_id}. Checking DB...")
        result = storage_manager.get_audio(job_id, f"{job_id}.mp3")
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")

    return Response(
        content=result,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"attachment; filename={job_id}.mp3"}
    )

@app.post("/cleanup")
async def cleanup_jobs():
    """Clean up old jobs across all services"""
    removed = 0
    for service in ServiceType:
        pattern = f"status:*:{service}"
        for key in redis_client.scan_iter(match=pattern):
            job_id = key.split(b":")[1].decode()  # Handle bytes key
            redis_client.delete(key)
            redis_client.delete(f"result:{job_id}:{service}")
            removed += 1
    return {"message": f"Removed {removed} old jobs"}
