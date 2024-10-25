# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

import json
import re


def find_json_objects(text):
    """
    Finds all substrings in the text that are potential JSON objects.

    Parameters:
    text (str): The input text to search for JSON objects.

    Returns:
    list: A list of strings, each a candidate JSON substring.
    """
    potential_jsons = []
    brace_stack = []
    start_idx = None

    for idx, char in enumerate(text):
        if char == "{":
            if not brace_stack:
                start_idx = idx
            brace_stack.append("{")
        elif char == "}":
            if brace_stack:
                brace_stack.pop()
                if not brace_stack and start_idx is not None:
                    end_idx = idx + 1
                    json_str = text[start_idx:end_idx]
                    potential_jsons.append(json_str)

    return potential_jsons


def find_json_arrays(text):
    """
    Finds all substrings in the text that are potential JSON arrays.

    Parameters:
    text (str): The input text to search for JSON arrays.

    Returns:
    list: A list of strings, each a candidate JSON array substring.
    """
    potential_jsons = []
    bracket_stack = []
    start_idx = None

    for idx, char in enumerate(text):
        if char == "[":
            if not bracket_stack:
                start_idx = idx
            bracket_stack.append("[")
        elif char == "]":
            if bracket_stack:
                bracket_stack.pop()
                if not bracket_stack and start_idx is not None:
                    end_idx = idx + 1
                    json_str = text[start_idx:end_idx]
                    potential_jsons.append(json_str)

    return potential_jsons


def extract_json(text):
    """
    Extracts the first valid JSON string found in the input text.

    The function handles JSON data that may be:
    - Enclosed in ```json ... ``` code blocks
    - Enclosed in ``` ... ``` code blocks
    - Directly present in the text without any enclosing

    Parameters:
    text (str): The input text containing JSON data.

    Returns:
    str: The JSON string if found and valid, otherwise None.
    """
    # Pattern to match code blocks enclosed in triple backticks, with or without 'json' qualifier
    code_block_pattern = r"```(?:json)?\s*(.*?)\s*```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)

    # Try to find JSON from code blocks first
    for block in code_blocks:
        json_str = block.strip()
        try:
            # Validate JSON string without parsing it into an object
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            continue

    # If no valid JSON in code blocks, search for JSON arrays and then JSON objects in the entire text
    potential_jsons = find_json_arrays(text) + find_json_objects(text)
    for json_str in potential_jsons:
        try:
            # Validate JSON string
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            continue

    # Return None if no valid JSON string is found
    return None
