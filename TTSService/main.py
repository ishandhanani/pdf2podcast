import json
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import logging
import edge_tts
import random
import concurrent.futures as cf
import io

logging.basicConfig(level=logging.INFO)

VOICE_LIST = [
    "en-US-AvaMultilingualNeural",
    "en-US-AndrewMultilingualNeural",
    "en-US-EmmaMultilingualNeural",
    "en-US-BrianMultilingualNeural",
]

def convert_text_to_mp3(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    with io.BytesIO() as file:
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
        return file.getvalue()
    
def process_edge_tts_request(json_input: str) -> str:
    # Parse the JSON input
    data = json.loads(json_input)
    dialogue = data.get('dialogue', [])

    logging.info(f"Processing TTS request with {len(dialogue)} dialogues")
    
    # Select random voices for the speakers
    voices = {"speaker-1": "en-US-AndrewMultilingualNeural", "speaker-2": "en-US-EmmaMultilingualNeural"}
    
    # Create a temp folder for the output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "sample.mp3"
        
        # Convert all dialogue entries to audio in parallel
        combined_audio = b""
        with cf.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    convert_text_to_mp3, 
                    entry.get('text', ''), 
                    voices[entry.get('speaker', 'speaker-1')]
                )
                for entry in dialogue
            ]
            for future in futures:
                combined_audio += future.result()
        
        # Write the combined audio to file
        with open(output_file, "wb") as f:
            f.write(combined_audio)
        
        # Copy to final location
        final_output = Path("sample.mp3")
        final_output.write_bytes(output_file.read_bytes())

    return str(final_output.absolute())


def process_f5_tts_request(json_input: str) -> str:
    # Parse the JSON input
    data = json.loads(json_input)
    dialogue = data.get('dialogue', [])

    logging.info(f"Processing TTS request with {len(dialogue)} dialogues")
    # Step 1: Create a temp folder
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 2 & 3: Convert dialogues and write to output.txt
        output_txt = Path(temp_dir) / "output.txt"
        with open(output_txt, "w") as f:
            for i, entry in enumerate(dialogue):
                speaker = entry.get('speaker', '')
                text = entry.get('text', '')
                if speaker == "speaker-1":
                    speaker = "main"
                elif speaker == "speaker-2":
                    speaker = "town"
                
                if i == 0:
                    f.write(f"{text}\n")
                else:
                    f.write(f"[{speaker}]{text}\n")

        # Step 4: Generate config.toml
        config_toml = Path(temp_dir) / "config.toml"
        with open(config_toml, "w") as f:
            f.write("""
# F5-TTS | E2-TTS
model = "F5-TTS"
ref_audio = "/workspace/F5-TTS/src/f5_tts/infer/examples/multi/main.flac"
# If an empty "", transcribes the reference audio automatically.
ref_text = ""
gen_text = ""
# File with text to generate. Ignores the text above.
gen_file = "output.txt"
remove_silence = true
output_dir = "tests"

[voices.main]
ref_audio = "/workspace/F5-TTS/src/f5_tts/infer/examples/multi/main.flac"
ref_text = ""

[voices.town]
ref_audio = "/workspace/F5-TTS/src/f5_tts/infer/examples/multi/town.flac"
ref_text = ""
            """.strip())

        logging.info(f"Running f5-tts_infer-cli with config.toml in {temp_dir}")
        # Step 5: Run f5-tts_infer-cli
        subprocess.run(["f5-tts_infer-cli", "-c", "config.toml"], cwd=temp_dir, check=True)

        # Step 6: Wait for subprocess to finish (implicit in the previous step)

        # Step 7: Run ffmpeg
    
        tests_dir = Path(temp_dir) / "tests"
        logging.info(f"Running ffmpeg in {tests_dir}")
        subprocess.run(["ffmpeg", "-i", "infer_cli_out.wav", "sample.mp3"], cwd=tests_dir, check=True)

        # Step 8: Copy the final output to a known location
        output_file = Path("sample.mp3")
        shutil.copy2(tests_dir / "sample.mp3", output_file)

    return str(output_file.absolute())

# FastAPI endpoint
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/generate_tts")
async def generate_tts(json_input: dict):
    try:
        mp3_path = process_edge_tts_request(json.dumps(json_input))
        return FileResponse(mp3_path, media_type="audio/mpeg", filename="sample.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "voices": VOICE_LIST  # List available voices
    }
