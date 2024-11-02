import requests
import os
import json
import time
from datetime import datetime

def test_combined_api():
    # API endpoint
    base_url = "http://localhost:8002"
    process_url = f"{base_url}/process_pdf"

    # Path to a sample PDF file for testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_pdf_path = os.path.join(current_dir, "sample.pdf")
    
    # Ensure the sample PDF file exists
    assert os.path.exists(sample_pdf_path), f"Sample PDF file not found at {sample_pdf_path}"

    # Prepare the payload
    transcription_params = {
        "duration": 20,
        "speaker_1_name": "Blackwell",
        "speaker_2_name": "Hopper",
        "model": "meta/llama-3.1-405b-instruct",
    }

    # Step 1: Submit the PDF file and get job ID
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Submitting PDF for processing...")
    with open(sample_pdf_path, "rb") as pdf_file:
        files = {"file": ("sample.pdf", pdf_file, "application/pdf")}
        response = requests.post(
            process_url,
            files=files,
            data={"transcription_params": json.dumps(transcription_params)}
        )

    # Check initial response
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    job_data = response.json()
    assert "job_id" in job_data, "Response missing job_id"
    job_id = job_data["job_id"]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Job ID received: {job_id}")

    # Step 2: Poll for results
    max_minutes = 20  # Maximum wait time in minutes
    poll_interval = 30  # Poll every 30 seconds
    max_attempts = (max_minutes * 60) // poll_interval
    attempts = 0
    
    start_time = time.time()
    last_status = None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Polling for results (will wait up to {max_minutes} minutes)...")
    while attempts < max_attempts:
        status_response = requests.get(f"{base_url}/status/{job_id}")
        assert status_response.status_code == 200, f"Status check failed with code {status_response.status_code}"

        # Check if we got the audio file
        if status_response.headers.get("content-type") == "audio/mpeg":
            elapsed_time = time.time() - start_time
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Audio file received after {elapsed_time:.1f} seconds")
            # Save the audio file
            with open("output.mp3", "wb") as f:
                f.write(status_response.content)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Audio file saved as 'output.mp3'")
            break

        # Otherwise, print status update (only if status has changed)
        status_data = status_response.json()
        current_status = f"{status_data['status']} - {status_data['message']}"
        
        if current_status != last_status:
            elapsed_time = time.time() - start_time
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed_time:.1f}s) Status: {current_status}")
            last_status = current_status

        if status_data["status"] == "failed":
            raise AssertionError(f"Job failed: {status_data['message']}")

        attempts += 1
        time.sleep(poll_interval)
    else:
        elapsed_time = time.time() - start_time
        raise TimeoutError(f"Test timed out after {elapsed_time:.1f} seconds ({max_minutes} minutes)")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Combined API test passed successfully.")

if __name__ == "__main__":
    test_combined_api()
