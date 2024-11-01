import requests
import os
import json

def test_combined_api():
    # API endpoint
    url = "http://localhost:8080/process_pdf"

    # Path to a sample PDF file for testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_pdf_path = os.path.join(current_dir, "sample.pdf")

    # Ensure the sample PDF file exists
    assert os.path.exists(sample_pdf_path), f"Sample PDF file not found at {sample_pdf_path}"

    # Prepare the payload
    transcription_params = {
        "duration": 20,
        "speaker_1_name": "Kate",
        "speaker_2_name": "Bob",
        "model": "mistralai/mistral-large-2-instruct"
    }

    # Open the PDF file and send it in the request
    with open(sample_pdf_path, "rb") as pdf_file:
        files = {"file": ("sample.pdf", pdf_file, "application/pdf")}
        response = requests.post(url, files=files, data={"transcription_params": json.dumps(transcription_params)})

    # Check if the request was successful
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"

    # Check if the response content is not empty
    assert response.content

    # Save the audio file
    with open("output.mp3", "wb") as f:
        f.write(response.content)

    print("Combined API test passed. Audio file saved as 'output.mp3'.")

if __name__ == "__main__":
    test_combined_api()
