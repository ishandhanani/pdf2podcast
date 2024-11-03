from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel
import time
from fastapi import HTTPException

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class StatusResponse(BaseModel):
    status: str
    message: str
    timestamp: float

class Job:
    def __init__(self):
        self.status = JobStatus.PENDING
        self.message = "Job started"
        self.timestamp = time.time()
        self.result: Optional[bytes] = None

class JobStatusManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}

    def create_job(self, job_id: str) -> Job:
        """Create a new job with the given ID"""
        self.jobs[job_id] = Job()
        return self.jobs[job_id]

    def get_job(self, job_id: str) -> Job:
        """Get a job by ID, raising HTTPException if not found"""
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return self.jobs[job_id]

    def update_status(self, job_id: str, status: JobStatus, message: str):
        """Update job status and message"""
        job = self.get_job(job_id)
        job.status = status
        job.message = message
        job.timestamp = time.time()

    def get_status(self, job_id: str) -> StatusResponse:
        """Get job status"""
        job = self.get_job(job_id)
        return StatusResponse(
            status=job.status.value,
            message=job.message,
            timestamp=job.timestamp
        )

    def set_result(self, job_id: str, result: bytes):
        """Set job result"""
        job = self.get_job(job_id)
        job.result = result

    def get_result(self, job_id: str) -> bytes:
        """Get job result, raising appropriate HTTPException if not ready"""
        job = self.get_job(job_id)
        
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(status_code=425, detail="Job not completed yet")
        
        if not job.result:
            raise HTTPException(status_code=500, detail="Job completed but no result found")
        
        return job.result

    def cleanup_old_jobs(self, max_age_seconds: int = 3600) -> int:
        """Remove jobs older than max_age_seconds, returns number of jobs removed"""
        current_time = time.time()
        removed = 0
        for job_id in list(self.jobs.keys()):
            if current_time - self.jobs[job_id].timestamp > max_age_seconds:
                del self.jobs[job_id]
                removed += 1
        return removed