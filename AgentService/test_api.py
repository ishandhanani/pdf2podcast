import requests
import json
import os

def test_transcribe_api():
    # API endpoint
    url = "http://localhost:8964/transcribe"

    # Load markdown content from file
    with open("sample.md", "r") as file:
        markdown_content = file.read()

    # Prepare the payload
    payload = {
        "markdown": markdown_content,
        "duration": 20,
        "speaker_1_name": "Donald Trump",
        "speaker_2_name": "Kamala Harris",
        "model": "meta/llama-3.1-405b-instruct"
    }

    # Send POST request
    response = requests.post(url, json=payload)

    # Check if the request was successful
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"

    # Parse the JSON response
    try:
        transcription = response.json()
    except json.JSONDecodeError:
        assert False, "Response is not valid JSON"

    print(transcription)


if __name__ == "__main__":
    test_transcribe_api()
    print("All tests passed!")

