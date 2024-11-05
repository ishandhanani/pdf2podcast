from fastapi import FastAPI, BackgroundTasks, HTTPException
from shared.shared_types import ServiceType, JobStatus
from shared.job import JobStatusManager
import flexagent as fa
from flexagent.backend import BackendConfig
from flexagent.engine import Value
from pydantic import BaseModel
from typing import List, Literal, Dict
import json
import os
import logging
from prompts import (
    RAW_OUTLINE_PROMPT,
    OUTLINE_PROMPT,
    SEGMENT_TRANSCRIPT_PROMPT,
    DEEP_DIVE_PROMPT,
    TRANSCRIPT_PROMPT,
    RAW_PODCAST_DIALOGUE_PROMPT_v2,
    FUSE_OUTLINE_PROMPT,
    REVISE_PROMPT,
    PODCAST_DIALOGUE_PROMPT
)
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DialogueEntry(BaseModel):
    text: str
    speaker: Literal["speaker-1", "speaker-2"]

class Conversation(BaseModel):
    scratchpad: str
    dialogue: List[DialogueEntry]

class PodcastSegment(BaseModel):
    section: str
    descriptions: List[str]
    duration: int

class PodcastOutline(BaseModel):
    title: str
    segments: List[PodcastSegment]

class TranscriptionRequest(BaseModel):
    markdown: str
    duration: int = 20
    speaker_1_name: str = "Bob" 
    speaker_2_name: str = "Kate"
    model: str = "meta/llama-3.1-405b-instruct"
    job_id: str

app = FastAPI(debug=True)
job_manager = JobStatusManager(ServiceType.AGENT)

def get_llm(model_name: str, backend_type: str = "nim"):
    # Use hardcoded base URL for 70B model, API_BASE env var for 405B
    api_base = "https://nim-pc8kmx5ae.brevlab.com/v1" if model_name == "meta/llama-3.1-70b-instruct" else os.getenv("API_BASE")
    
    backend = BackendConfig(
        backend_type=backend_type,
        model_name=model_name,
        api_key=os.getenv("NIM_KEY"),
        api_base=api_base
    )
    return fa.ops.LLM().to(backend)

def retry_nim_request(llm, messages, retries=5, sync=True, json_schema=None):
    for _ in range(retries):
        try:
            extra_body = None
            if json_schema:
                extra_body = {
                    "nvext": {
                        "guided_json": json_schema
                    }
                }
            if sync:
                return llm(messages, extra_body=extra_body).get()
            else:
                return llm(messages, extra_body=extra_body)
        except Exception as e:
            logger.error(f"Failed to get response: {e}")
            time.sleep(3)
    raise Exception("Failed to get response")

def process_transcription(job_id: str, request: TranscriptionRequest):
    try:
        # Initialize LLMs
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Initializing LLMs")
        reasoning_llm = get_llm(request.model)
        json_llm = get_llm("meta/llama-3.1-70b-instruct")
        
        schema = PodcastOutline.model_json_schema()

        # Get raw outline using 405B (complex reasoning)
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Generating initial outline")
        prompt = RAW_OUTLINE_PROMPT.render(text=request.markdown, duration=request.duration)
        messages = [{"role": "user", "content": prompt}]
        raw_outline = retry_nim_request(reasoning_llm, messages)

        # Process outline using 70B (JSON formatting)
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Converting raw outline to structured format")
        prompt = OUTLINE_PROMPT.render(text=raw_outline, schema=json.dumps(schema, indent=2))
        messages = [{"role": "user", "content": prompt}]
        outline = retry_nim_request(json_llm, messages, json_schema=schema)
        outline_json = json.loads(outline)
        
        # Process segments
        longest_segment_idx = max(range(len(outline_json["segments"])), 
                                key=lambda i: outline_json["segments"][i]["duration"])

        segments = []
        sub_outline = {}
        for idx, segment in enumerate(outline_json["segments"]):
            job_manager.update_status(
                job_id, 
                JobStatus.PROCESSING, 
                f"Processing segment {idx + 1}/{len(outline_json['segments'])}: {segment['section']}"
            )
            
            if idx == longest_segment_idx:
                ret = deep_dive_segment(job_id, request.markdown, segment, reasoning_llm, json_llm, schema)
                segments.append(ret[0])
                sub_outline = ret[1]
            else:
                # Complex content generation using 405B
                prompt = SEGMENT_TRANSCRIPT_PROMPT.render(
                    text=request.markdown,
                    duration=segment["duration"],
                    topic=segment["section"],
                    angles="\n".join(segment["descriptions"])
                )
                messages = [{"role": "user", "content": prompt}]
                segments.append(retry_nim_request(reasoning_llm, messages, sync=False))

        # Process dialogue using 405B (complex content generation)
        segment_transcripts = []
        for idx, segment in enumerate(outline_json["segments"]):
            job_manager.update_status(
                job_id,
                JobStatus.PROCESSING,
                f"Converting segment {idx + 1}/{len(outline_json['segments'])} to dialogue"
            )
            prompt = RAW_PODCAST_DIALOGUE_PROMPT_v2.render(
                text=segments[idx].get(),
                duration=segment["duration"],
                descriptions=segment["descriptions"],
                speaker_1_name=request.speaker_1_name,
                speaker_2_name=request.speaker_2_name
            )
            messages = [{"role": "user", "content": prompt}]
            segment_transcripts.append(retry_nim_request(reasoning_llm, messages, sync=False))

        # Combine transcripts
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Combining segments")
        full_transcript = "\n".join([segment.get() for segment in segments])
        conversation = "\n".join([segment.get() for segment in segment_transcripts])

        # Fuse outline using 405B (complex reasoning)
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Fusing outline")
        prompt = FUSE_OUTLINE_PROMPT.render(overall_outline=outline, sub_outline=sub_outline)
        messages = [{"role": "user", "content": prompt}]
        full_outline = retry_nim_request(reasoning_llm, messages)

        # Revise dialogue using 405B (complex content generation)
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Revising dialogue")
        prompt = REVISE_PROMPT.render(
            raw_transcript=full_transcript,
            dialogue_transcript=conversation,
            outline=full_outline,
        )
        messages = [{"role": "user", "content": prompt}]
        conversation = retry_nim_request(reasoning_llm, messages)

        # Convert to JSON using 70B
        schema = Conversation.model_json_schema()
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Converting to final format")
        prompt = PODCAST_DIALOGUE_PROMPT.render(
            text=conversation,
            schema=json.dumps(schema, indent=2),
            speaker_1_name=request.speaker_1_name,
            speaker_2_name=request.speaker_2_name,
        )
        messages = [{"role": "user", "content": prompt}]
        final_conversation = retry_nim_request(json_llm, messages, json_schema=schema)
        
        # Store result
        result = json.loads(final_conversation)
        job_manager.set_result(job_id, json.dumps(result).encode())
        job_manager.update_status(job_id, JobStatus.COMPLETED, "Transcription completed successfully")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        job_manager.update_status(job_id, JobStatus.FAILED, str(e))
        raise

def deep_dive_segment(job_id: str, text: str, segment: Dict[str, str], reasoning_llm, json_llm, schema):
    status_msg = f"Performing deep dive analysis of segment: {segment['section']}"
    job_manager.update_status(job_id, JobStatus.PROCESSING, status_msg)
    logger.info(f"Job {job_id}: {status_msg}")
    
    # Complex reasoning using 405B
    prompt = DEEP_DIVE_PROMPT.render(
        text=text,
        topic=segment["descriptions"],
        duration=segment["duration"]
    )
    messages = [{"role": "user", "content": prompt}]
    outline = retry_nim_request(reasoning_llm, messages)

    # JSON formatting using 70B
    prompt = OUTLINE_PROMPT.render(
        text=outline,
        schema=json.dumps(schema, indent=2)
    )
    messages = [{"role": "user", "content": prompt}]
    outline_json = json.loads(
        retry_nim_request(json_llm, messages, json_schema=schema)
    )
    
    segments = []
    for subsegment in outline_json["segments"]:
        job_manager.update_status(
            job_id,
            JobStatus.PROCESSING,
            f"Processing subsegment: {subsegment['section']}"
        )
        prompt = SEGMENT_TRANSCRIPT_PROMPT.render(
            text=text,
            duration=subsegment["duration"],
            topic=subsegment["section"],
            angles="\n".join(subsegment["descriptions"])
        )
        messages = [{"role": "user", "content": prompt}]
        segments.append(retry_nim_request(reasoning_llm, messages, sync=False))
    
    texts = [segment.get() for segment in segments]
    return (Value("\n".join(texts)), outline_json)

@app.post("/transcribe", status_code=202)
def transcribe(request: TranscriptionRequest, background_tasks: BackgroundTasks):
    job_manager.create_job(request.job_id)
    background_tasks.add_task(process_transcription, request.job_id, request)
    return {"job_id": request.job_id}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    return job_manager.get_status(job_id)

@app.get("/output/{job_id}")
def get_output(job_id: str):
    result = job_manager.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return json.loads(result.decode())  # Decode bytes to string before JSON parsing

@app.get("/transcribe/health")
def health():
    return {
        "status": "healthy",
        "version": fa.__version__,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8964)