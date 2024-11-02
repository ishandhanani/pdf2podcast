import jinja2
import flexagent as fa
from dotenv import load_dotenv
from flexagent.backend import BackendConfig
from flexagent.engine import Value
from pydantic import BaseModel
from typing import List, Literal, Dict
import json
import os
import time
import logging
logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

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




RAW_OUTLINE_PROMPT = jinja2.Template("""
I want to make the following paper into a podcast transcript for {{ duration }} minutes, to help audience understand background, innovation, impact and future work. 

Come up the structure of the podcast.
                                 
{{ text }}
                                     
Innovation should be the focus of the podcast, and the most important part of the podcast, with enough details.
                          
""")

OUTLINE_PROMPT = jinja2.Template("""
Given the free form outline, convert in into a structured outline without losing any information.                                 

{{ text }}
                                                           
The result must conform to the following JSON schema:\n```json\n{{ schema }}\n```\n\n
Provide the result enclosed in triple backticks with 'json' on the first line.                            
""")


SEGMENT_TRANSCRIPT_PROMPT = jinja2.Template("""
Make a transcript given the text:

{{ text }}
                                            
The transcript is about {{ duration }} minutes, approximately {{ (duration * 180) | int }} words.
The transcript's subject is {{ topic }}, and should focus on the following topics: {{ angles }}
                                            
Explain all concepts clearly, assuming no prior knowledge
Use analogies, stories, or examples to illustrate points
Address potential questions or counterarguments
Provide context and background information throughout
Make sure the details, numbers are accurate and comprehensive
                                            
Dive deep into each topic, and provide enough details given the time budget, don't leave any stone unturned.
                                         
""")


DEEP_DIVE_PROMPT = jinja2.Template("""
You will be given some content, short ideas or thoughts about the content.

Your task is to expand the content into a detailed and comprehensive explanation, with enough details and examples.

Here is the content

{{text}}
                                   


The topic will be around
                                   
{{topic}}
                                   
Dive deep into each topic, come up with an outline with topics and subtopics to help fully understand the content.
Expand the topics, don't add any other topics. Allocate time budget for each topic. Total time budget should be {{ duration }} minutes.
Focus on the most important topics and ideas, and allocate more time budget to them.
Avoid introduction and conclusion in the outline, focus on expanding into subtopics.
""")

TRANSCRIPT_PROMPT = jinja2.Template("""
Given the transcript of different segments,combine and optimize the transcript to make the flow more natural.
The content should be strictly following the transcript, and only optimize the flow. Keep all the details, and storytelling contents.
For each segment, there is also a time budget for reference.

{% for segment, duration in segments %}

Time budget: {{ duration }} minutes, approximately {{ (duration * 180) | int }} words.
{{ segment }}                                    

{% endfor %}
                                    
Only return the full transcript, no need to include any other information like time budget or segment name.
""")




RAW_PODCAST_DIALOGUE_PROMPT_v2 = jinja2.Template("""
Your task is to transform the provided input transcript into a lively, engaging, and informative podcast dialogue. 

There are two speakers, speaker-1 and speaker-2.
speaker-1's name is {{ speaker_1_name }}, and speaker-2's name is {{ speaker_2_name }}.

Given the following conversation, make introductions for both speakers at beginning of the conversation.
During the conversation, occasionally mention the speaker's name to refer to them, to make the conversation more natural.
Incorporate natural speech patterns, including occasional verbal fillers (e.g., "um," "well," "you know")
Use casual language and ensure the dialogue flows smoothly, reflecting a real-life conversation
The fillers should be used naturally, not in every sentence, and not in a robotic way but related to topic and conversation context.
                                          
Maintain a lively pace with a mix of serious discussion and lighter moments
Use rhetorical questions or hypotheticals to involve the listener
Create natural moments of reflection or emphasis
     
Allow for natural interruptions and back-and-forth between host and guest
Ensure the guest's responses are substantiated by the input text, avoiding unsupported claims                                   
Avoid long sentences from either speaker, break them into conversations between two speakers.
Throughout the script, strive for authenticity in the conversation. Include:
   - Moments of genuine curiosity or surprise from the host
   - Instances where the guest might briefly struggle to articulate a complex idea
   - Light-hearted moments or humor when appropriate
   - Brief personal anecdotes or examples that relate to the topic (within the bounds of the input text)
                 
Don't lose any information or details in the transcript. It is only format conversion, so strictly follow the transcript.
                                                 
This segment is about {{ duration }} minutes, approximately {{ (duration * 180) | int }} words.
The topic is {{ descriptions }}
                                          
You should keep all analogies, stories, examples, and quotes from the transcript.

Here is the transcript:
{{text}}
                                          
Only return the full dialogue transcript, no need to include any other information like time budget or segment name.
Don't add introduction and ending to the dialogue unless it is provided in the transcript.
                                                                  
""")


FUSE_OUTLINE_PROMPT = jinja2.Template("""
You are given two outlines, one is overall outline, another is sub-outline for one section in the overall outline.
You need to fuse the two outlines into a new outline, to represent the whole podcast without losing any descriptions in sub sections.
Ignore the time budget in the sub-outline, and use the time budget in the overall outline.
Overall outline:
{{ overall_outline }}

Sub-outline:
{{ sub_outline }}

Output the new outline with the tree structure.

""")


REVISE_PROMPT = jinja2.Template("""
You are given a podcast dialogue transcript, and a raw transcript of the podcast.
You are only allowed to copy information from the raw dialogue transcript to make the conversation more natural and engaging, but exactly follow the outline.
                                
Outline:
{{ outline}}


Here is the dialogue transcript:
{{ dialogue_transcript }}

You need also to break long sentences from either speaker into conversations between two speakers, by inserting more dialogue entries and verbal fillers (e.g., "um")
Don't let a single speaker talk more than 2 sentences, and break the conversation into multiple exchanges between two speakers.
                                
Don't make any explict transition between sections, this is one podcast, and the sections are connected.
Don't use words like "Welcome back" or "Now we are going to talk about" etc.
Don't make introductions in the middle of the conversation.
Merge related topics according to outline and don't repeat same things in different place.
                                
Don't lose any information or details from the raw transcript, only make the conversation flow more natural.
""")

PODCAST_DIALOGUE_PROMPT = jinja2.Template("""
Given a podcast transcript between two speakers, convert it into a structured JSON format.
- Only do conversion
- Don't miss any information in the transcript

There are two speakers, speaker-1 and speaker-2.
speaker-1's name is {{ speaker_1_name }}, and speaker-2's name is {{ speaker_2_name }}.
                                          
Here is the original transcript:
{{ text }}
                                          
The result must conform to the following JSON schema:\n```json\n{{ schema }}\n```\n\n
Provide the result enclosed in triple backticks with 'json' on the first line.
""")


def retry_nim_request(llm, messages, retries=5, sync=True):
    for _ in range(retries):
        try:
            if sync:
                return llm(messages).get()
            else:
                return llm(messages)
        except Exception as e:
            logging.error(f"Failed to get response: {e}")
        time.sleep(3)
    raise Exception("Failed to get response")

def deep_dive_agent(text: str, segment: Dict[str, str], llm):
    logging.info(f"Deep diving into topic: {segment['section']}")
    schema = PodcastOutline.model_json_schema()
    # raw outline
    prompt = DEEP_DIVE_PROMPT.render(text=text, topic=segment["descriptions"], duration=segment["duration"])
    messages = [{"role": "user", "content": prompt}]
    outline =retry_nim_request(llm, messages)

    # json outline
    prompt = OUTLINE_PROMPT.render(text=outline, schema=json.dumps(schema, indent=2))
    messages = [{"role": "user", "content": prompt}]
    outline_json = json.loads(fa.utils.json_func.extract_json(retry_nim_request(llm, messages)))
    
    # print(outline_json)
    # explore each topic
    segments = []
    try:
        for segment in outline_json["segments"]:
            logging.info(f"Getting subsegment: {segment['section']}")
            prompt = SEGMENT_TRANSCRIPT_PROMPT.render(text=text, duration=segment["duration"], topic=segment["section"], angles="\n".join(segment["descriptions"]))
            messages = [{"role": "user", "content": prompt}]
            segments.append(retry_nim_request(llm, messages, sync=False))
    except Exception as e:
        logging.error(f"Failed to get segments: {e}")
        raise e
    
    texts = [segment.get() for segment in segments]
    return (Value("\n".join(texts)), outline_json)
    
def transcript_agent(text: str,
                    duration: int = 20,
                    speaker_1_name: str = "Donald Trump",
                    speaker_2_name: str = "Kamala Harris",
                    model: str = "meta/llama-3.1-405b-instruct",
                    api_key: str = None):  # Add api_key parameter
    schema = PodcastOutline.model_json_schema()
    backend = BackendConfig(
        backend_type="nim",
        model_name=model,
        api_key=api_key,  # Use provided API key instead of env var
    )
    llm = fa.ops.LLM().to(backend)

    # raw outline
    try:
        logging.info(f"Getting raw outline")
        prompt = RAW_OUTLINE_PROMPT.render(text=text, duration=duration)
        messages = [{"role": "user", "content": prompt}]
        raw_outline = retry_nim_request(llm, messages)
    except Exception as e:
        logging.error(f"Failed to get raw outline: {e}")
        raise e
    

    # # outline
    try:
        logging.info(f"Getting outline")
        prompt = OUTLINE_PROMPT.render(text=raw_outline, schema=json.dumps(schema, indent=2))
        messages = [{"role": "user", "content": prompt}]
        outline = retry_nim_request(llm, messages)
        logging.info(f"Got outline: {outline}")
    except Exception as e:
        logging.error(f"Failed to get outline: {e}")
        raise e

    outline_json = json.loads(fa.utils.json_func.extract_json(outline))


    
    longest_segment_idx = max(range(len(outline_json["segments"])), 
                            key=lambda i: outline_json["segments"][i]["duration"])

    segments = []
    sub_outline = {}
    try:
        for idx, segment in enumerate(outline_json["segments"]):
            logging.info(f"Getting segment: {segment['section']}")
            if idx == longest_segment_idx:
                ret = deep_dive_agent(text, segment, llm)
                segments.append(ret[0])
                sub_outline = ret[1]
            else:
                prompt = SEGMENT_TRANSCRIPT_PROMPT.render(text=text, duration=segment["duration"], topic=segment["section"], angles="\n".join(segment["descriptions"]))
                messages = [{"role": "user", "content": prompt}]
                segments.append(retry_nim_request(llm, messages, sync=False))
    except Exception as e:
        logging.error(f"Failed to get segments: {e}")
        raise e
    
    # # Now convert the full transcript into a podcast dialogue
    
    

     ## v2 directly to podcast dialogue per segment
    segment_transcripts = []
    for idx, segment in enumerate(outline_json["segments"]):
        prompt = RAW_PODCAST_DIALOGUE_PROMPT_v2.render(text=segments[idx].get(), duration=segment["duration"], descriptions=segment["descriptions"], speaker_1_name=speaker_1_name, speaker_2_name=speaker_2_name)
        messages = [{"role": "user", "content": prompt}]
        segment_transcripts.append(retry_nim_request(llm, messages, sync=False))

    full_transcript = "\n".join([segment.get() for segment in segments])
    
    conversation = "\n".join([segment.get() for segment in segment_transcripts])

    # Fuse outline
    logging.info(f"Fusing outline")
    try:
        prompt = FUSE_OUTLINE_PROMPT.render(overall_outline=outline, sub_outline=sub_outline)
        messages = [{"role": "user", "content": prompt}]
        full_outline = retry_nim_request(llm, messages)
    except Exception as e:
        logging.error(f"Failed to fuse outline: {e}")
        raise e

    schema = Conversation.model_json_schema()


    ## Add a revise loop to add missing information

    try:
        logging.info(f"Revising podcast dialogue")
        prompt = REVISE_PROMPT.render(
            raw_transcript=full_transcript,
            dialogue_transcript=conversation,
            outline=full_outline,
        )
        messages = [{"role": "user", "content": prompt}]
        conversation = retry_nim_request(llm, messages)
    except Exception as e:
        logging.error(f"Failed to revise podcast dialogue: {e}")
        raise e
    
    with open("revised_conversation.md", "w") as f:
        f.write(conversation)

    # convert to podcast dialogue in JSON format
    try:
        logging.info(f"Converting to podcast dialogue")
        logging.info(f"Rendering prompt now")
        prompt = PODCAST_DIALOGUE_PROMPT.render(
            text=conversation,
            schema=json.dumps(schema, indent=2),
            speaker_1_name=speaker_1_name,
            speaker_2_name=speaker_2_name,
        )
        messages = [{"role": "user", "content": prompt}]
        logging.info(f"Trying the nim request.")
        conversation = retry_nim_request(llm, messages)
        logging.info(f"Finished trying the nim request.")
    except Exception as e:
        logging.error(f"Failed to convert to podcast dialogue: {e}")
        raise e
    
    return fa.utils.json_func.extract_json(conversation)

class TranscriptionRequest(BaseModel):
    markdown: str
    duration: int = 20
    speaker_1_name: str = "Bob" 
    speaker_2_name: str = "Kate"
    model: str = "meta/llama-3.1-405b-instruct"
    api_key: str

@app.post("/transcribe")
async def transcribe(request: TranscriptionRequest):
    try:
        result = transcript_agent(
            text=request.markdown,
            duration=request.duration,
            speaker_1_name=request.speaker_1_name,
            speaker_2_name=request.speaker_2_name,
            model=request.model,
            api_key=request.api_key
        )
        try:
            return json.loads(result)
        except Exception as e:
            raise HTTPException(status_code=503, detail="The JSON part failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add at the beginning with other routes

@app.get("/transcribe/health")
async def health():
    return {
        "status": "healthy",
        "version": fa.__version__,  # Return flexagent version
    }

if __name__ == "__main__":
    load_dotenv()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8964)