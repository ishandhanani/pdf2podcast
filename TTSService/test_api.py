import requests
import json

def test_tts_api():
    # API endpoint URL
    url = "http://localhost:8888/generate_tts"

    # Load sample JSON data
    with open("sample.json", "r") as f:
        json_data = json.load(f)

    # Make POST request to the API
    response = requests.post(url, json=json_data)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the MP3 file
        with open("output.mp3", "wb") as f:
            f.write(response.content)
        print("TTS generation successful. MP3 file saved as 'output.mp3'.")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_tts_api()

