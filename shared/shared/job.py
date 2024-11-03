from shared.shared_types import ServiceType, JobStatus  # Import JobStatus as well
import redis
import time
import json
import threading

class JobStatusManager:
    def __init__(self, service_type: ServiceType, redis_url="redis://redis:6379"):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.service_type = service_type
        self._lock = threading.Lock()

    def create_job(self, job_id: str):
        update = {
            "job_id": job_id,  # Add job_id
            "status": JobStatus.PENDING.value,  # Use enum value
            "message": "Job created",
            "service": self.service_type.value,  # Convert enum to string value
            "timestamp": time.time()
        }
        self.redis.hset(f"status:{job_id}:{self.service_type}", mapping=update)
        self.redis.publish("status_updates:all", json.dumps(update))

    def update_status(self, job_id: str, status: JobStatus, message: str):
        update = {
            "job_id": job_id,  # Add job_id
            "status": status.value,  # Use enum value
            "message": message,
            "service": self.service_type.value,  # Convert enum to string value
            "timestamp": time.time()
        }
        self.redis.hset(f"status:{job_id}:{self.service_type}", mapping=update)
        self.redis.publish("status_updates:all", json.dumps(update))

    def set_result(self, job_id: str, result: bytes):
        self.redis.set(f"result:{job_id}:{self.service_type}", result)

    def get_result(self, job_id: str):
        result = self.redis.get(f"result:{job_id}:{self.service_type}")
        return result if result else None

    def get_status(self, job_id: str):
        status = self.redis.hgetall(f"status:{job_id}:{self.service_type}")
        if not status:
            raise ValueError("Job not found")
        return status

    def cleanup_old_jobs(self, max_age=3600):
        current_time = time.time()
        removed = 0
        pattern = f"status:*:{self.service_type}"
        for key in self.redis.scan_iter(match=pattern):
            status = self.redis.hgetall(key)
            if float(status.get("timestamp", 0)) < current_time - max_age:
                self.redis.delete(key)
                job_id = key.split(":")[1]
                self.redis.delete(f"result:{job_id}:{self.service_type}")
                removed += 1
        return removed