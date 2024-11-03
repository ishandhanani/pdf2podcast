from pydantic import BaseModel
from typing import Optional
from enum import Enum

class ServiceType(str, Enum):
    PDF = "pdf"
    AGENT = "agent"
    TTS = "tts"

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class StatusUpdate(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    service: ServiceType
    timestamp: float
    data: Optional[dict] = None