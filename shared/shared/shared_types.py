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
    message: Optional[str] = None
    service: Optional[ServiceType] = None
    timestamp: Optional[float] = None
    data: Optional[dict] = None

class StatusResponse(BaseModel):
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None