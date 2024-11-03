import requests
import os
import json
import time
import redis
from datetime import datetime
from threading import Thread, Event

def get_time():
    return datetime.now().strftime('%H:%M:%S')

class StatusListener:
    def __init__(self, job_id):
        self.job_id = job_id
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        self.stop_event = Event()
        self.services = {'pdf', 'agent', 'tts'}
        self.last_statuses = {service: None for service in self.services}
        self.tts_completed = Event()
        
    def start(self):
        self.pubsub.subscribe('status_updates:all')
        self.thread = Thread(target=self._listen)
        self.thread.start()
        
    def stop(self):
        self.stop_event.set()
        self.pubsub.unsubscribe()
        self.thread.join()
        self.redis_client.close()
        
    def _listen(self):
        start_time = time.time()
        
        for message in self.pubsub.listen():
            if self.stop_event.is_set():
                break
                
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    # Only process messages for our job
                    if data.get('job_id') == self.job_id:
                        service = data.get('service')
                        status = data.get('status')
                        msg = data.get('message', '')
                        
                        if service in self.services:
                            current_status = f"{service}: {status} - {msg}"
                            if current_status != self.last_statuses[service]:
                                elapsed = time.time() - start_time
                                print(f"[{get_time()}] ({elapsed:.1f}s) {current_status}")
                                self.last_statuses[service] = current_status
                                
                                if status == 'failed':
                                    print(f"[{get_time()}] Job failed in {service}: {msg}")
                                    self.stop_event.set()
                                    
                                if service == 'tts' and status == 'completed':
                                    self.tts_completed.set()
                                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing message: {e}")
                    continue

def test_api():
    # API endpoint
    base_url = "http://localhost:8002"
    process_url = f"{base_url}/process_pdf"

    # Path to a sample PDF file for testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_pdf_path = os.path.join(current_dir, "PNP_Proof.pdf")
    
    # Ensure the sample PDF file exists
    assert os.path.exists(sample_pdf_path), f"Sample PDF file not found at {sample_pdf_path}"
    
    # Prepare the payload
    transcription_params = {
        "duration": 5,
        "speaker_1_name": "Blackwell",
        "speaker_2_name": "Hopper",
        "model": "nvidia/llama-3.1-nemotron-51b-instruct",
    }

    # Step 1: Submit the PDF file and get job ID
    print(f"\n[{get_time()}] Submitting PDF for processing...")
    with open(sample_pdf_path, "rb") as pdf_file:
        files = {"file": ("sample.pdf", pdf_file, "application/pdf")}
        response = requests.post(
            process_url,
            files=files,
            data={"transcription_params": json.dumps(transcription_params)}
        )

    # Check initial response
    assert response.status_code == 202, f"Expected status code 202, but got {response.status_code}"
    job_data = response.json()
    assert "job_id" in job_data, "Response missing job_id"
    job_id = job_data["job_id"]
    print(f"[{get_time()}] Job ID received: {job_id}")

    # Step 2: Start listening for status updates
    listener = StatusListener(job_id)
    listener.start()
    
    try:
        # Wait for TTS completion or timeout
        max_wait = 20 * 60  # 20 minutes in seconds
        if not listener.tts_completed.wait(timeout=max_wait):
            raise TimeoutError(f"Test timed out after {max_wait} seconds")
            
        # If we get here, TTS completed successfully
        print(f"\n[{get_time()}] TTS processing completed, retrieving audio file...")
        
        # Get the final output
        output_response = requests.get(f"{base_url}/output/{job_id}")
        assert output_response.status_code == 200, f"Failed to get output, status code: {output_response.status_code}"
        
        # Save the audio file
        with open("output.mp3", "wb") as f:
            f.write(output_response.content)
        print(f"[{get_time()}] Audio file saved as 'output.mp3'")
        
    finally:
        listener.stop()

if __name__ == "__main__":
    test_api()