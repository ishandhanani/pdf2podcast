from fastapi import FastAPI, BackgroundTasks
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

def retry_nim_request(llm, messages, retries=5, sync=True):
    for _ in range(retries):
        try:
            if sync:
                return llm(messages).get()
            else:
                return llm(messages)
        except Exception as e:
            logger.error(f"Failed to get response: {e}")
            time.sleep(3)
    raise Exception("Failed to get response")

def process_transcription(job_id: str, request: TranscriptionRequest):
    try:
        # Initialize LLM
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Initializing LLM")
        schema = PodcastOutline.model_json_schema()
        backend = BackendConfig(
            backend_type="nim",
            model_name=request.model,
            api_key=os.getenv("NIM_KEY"),
            api_base="https://405b-pg7podjpv.brevlab.com/v1"
        )
        llm = fa.ops.LLM().to(backend)

        # Get raw outline
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Generating initial outline")
        prompt = RAW_OUTLINE_PROMPT.render(text=request.markdown, duration=request.duration)
        messages = [{"role": "user", "content": prompt}]
        raw_outline = retry_nim_request(llm, messages)

        # Process outline
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Processing outline")
        prompt = OUTLINE_PROMPT.render(text=raw_outline, schema=json.dumps(schema, indent=2))
        messages = [{"role": "user", "content": prompt}]
        outline = retry_nim_request(llm, messages)
        outline_json = json.loads(fa.utils.json_func.extract_json(outline))

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
                ret = deep_dive_segment(job_id, request.markdown, segment, llm)
                segments.append(ret[0])
                sub_outline = ret[1]
            else:
                prompt = SEGMENT_TRANSCRIPT_PROMPT.render(
                    text=request.markdown,
                    duration=segment["duration"],
                    topic=segment["section"],
                    angles="\n".join(segment["descriptions"])
                )
                messages = [{"role": "user", "content": prompt}]
                segments.append(retry_nim_request(llm, messages, sync=False))

        # Process dialogue
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
            segment_transcripts.append(retry_nim_request(llm, messages, sync=False))

        # Combine transcripts
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Combining segments")
        full_transcript = "\n".join([segment.get() for segment in segments])
        conversation = "\n".join([segment.get() for segment in segment_transcripts])

        # Fuse outline
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Fusing outline")
        prompt = FUSE_OUTLINE_PROMPT.render(overall_outline=outline, sub_outline=sub_outline)
        messages = [{"role": "user", "content": prompt}]
        full_outline = retry_nim_request(llm, messages)

        # Revise dialogue
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Revising dialogue")
        prompt = REVISE_PROMPT.render(
            raw_transcript=full_transcript,
            dialogue_transcript=conversation,
            outline=full_outline,
        )
        messages = [{"role": "user", "content": prompt}]
        conversation = retry_nim_request(llm, messages)

        # Convert to JSON
        schema = Conversation.model_json_schema()
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Converting to final format")
        prompt = PODCAST_DIALOGUE_PROMPT.render(
            text=conversation,
            schema=json.dumps(schema, indent=2),
            speaker_1_name=request.speaker_1_name,
            speaker_2_name=request.speaker_2_name,
        )
        messages = [{"role": "user", "content": prompt}]
        final_conversation = retry_nim_request(llm, messages)
        
        # Store result
        result = json.loads(fa.utils.json_func.extract_json(final_conversation))
        job_manager.set_result(job_id, json.dumps(result).encode())
        job_manager.update_status(job_id, JobStatus.COMPLETED, "Transcription completed successfully")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        job_manager.update_status(job_id, JobStatus.FAILED, str(e))
        raise

def deep_dive_segment(job_id: str, text: str, segment: Dict[str, str], llm):
    status_msg = f"Performing deep dive analysis of segment: {segment['section']}"
    job_manager.update_status(job_id, JobStatus.PROCESSING, status_msg)
    logger.info(f"Job {job_id}: {status_msg}")
    
    schema = PodcastOutline.model_json_schema()
    prompt = DEEP_DIVE_PROMPT.render(
        text=text,
        topic=segment["descriptions"],
        duration=segment["duration"]
    )
    messages = [{"role": "user", "content": prompt}]
    outline = retry_nim_request(llm, messages)

    prompt = OUTLINE_PROMPT.render(
        text=outline,
        schema=json.dumps(schema, indent=2)
    )
    messages = [{"role": "user", "content": prompt}]
    outline_json = json.loads(fa.utils.json_func.extract_json(retry_nim_request(llm, messages)))
    
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
        segments.append(retry_nim_request(llm, messages, sync=False))
    
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
    return json.loads(result)

@app.get("/transcribe/health")
def health():
    return {
        "status": "healthy",
        "version": fa.__version__,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8964)