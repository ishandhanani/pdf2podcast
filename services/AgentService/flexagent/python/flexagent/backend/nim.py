# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
from typing import Dict, List, Optional, Any

from .openai import OpenAIBackend
from .registry import reg
from .sampling_params import SamplingParams
import logging
logging.basicConfig(level=logging.INFO)

class NIMBackend(OpenAIBackend):
    """
    A backend class for interacting with NVIDIA Inference Microservices (NIM) API.

    This class inherits from OpenAIBackend and customizes it for use with NIM.
    It provides an interface for generating text using NIM's API, which is
    compatible with the OpenAI API format.

    :param model_name: The name of the NIM model to use.
    :param api_key: The API key for authenticating with NIM. If not provided,
                    it will attempt to use the NIM_KEY environment variable.
    :param api_base: The base URL for the NIM API. Defaults to the standard NIM endpoint.

    :type model_name: str
    :type api_key: Optional[str]
    :type api_base: str

    :raises ValueError: If no API key is provided or found in the environment.

    Example:
        >>> from flexagent.backend.nim import NIMBackend
        >>> from flexagent.backend.sampling_params import SamplingParams
        >>>
        >>> # Initialize the NIM backend
        >>> nim_backend = NIMBackend(model_name="meta/llama-3.1-8b-instruct")
        >>>
        >>> # Set up prompts and parameters
        >>> system_prompt = "You are a helpful assistant."
        >>> user_prompt = "What is the capital of France?"
        >>> sampling_params = SamplingParams(max_new_tokens=50, temperature=0.7)
        >>>
        >>> # Generate a response
        >>> response = nim_backend.generate(system_prompt, user_prompt, sampling_params)
        >>> print(response)
        The capital of France is Paris.
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("NIM_KEY")
        self.max_retry = 5
        if not api_key:
            raise ValueError(
                "NIM API key is required. Set it as NIM_KEY environment variable or pass it explicitly."
            )

        super().__init__(model_name, api_key=api_key, api_base=api_base)

    def generate(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[SamplingParams] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate text using the NIM API based on the given prompts and parameters.

        This method overrides the parent class method to allow for any NIM-specific
        adjustments that might be needed in the future.

        :param messages: The messages to generate text from.
        :param sampling_params: Optional parameters to control the text generation.

        :type messages: List[Dict[str, str]]
        :type sampling_params: Optional[SamplingParams]

        :return: The generated text response.
        :rtype: str

        Example:
            >>> nim_backend = NIMBackend(model_name="meta/llama-3.1-8b-instruct")
            >>> system_prompt = "You are a helpful assistant."
            >>> user_prompt = "Explain the concept of machine learning in one sentence."
            >>> params = SamplingParams(max_new_tokens=30, temperature=0.8)
            >>> response = nim_backend.generate(system_prompt, user_prompt, params)
            >>> print(response)
            Machine learning is the process of training algorithms to learn patterns
            from data and make predictions or decisions without explicit programming.
        """
        # NIM-specific adjustments can be made here if needed in the future
        for _ in range(self.max_retry):
            try:
                return super().generate(messages, sampling_params, extra_body=extra_body)
            except Exception as e:
                logging.error(f"Error generating text: {e}")
        raise Exception("Failed to generate text after multiple retries")


@reg("nim")
def get_nim_backend(
    model_name: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> NIMBackend:
    """
    Factory function to create an NIMBackend instance.

    This function is registered with the "nim" key and can be used to create
    OpenAIBackend instances.

    :param model_name: The name of the NIM model to use.
    :param api_key: The API key for authentication with NIM.
    :param api_base: Optional custom API base URL.
    :return: An instance of NIMBackend.

    :example:

    >>> backend = get_nim_backend("meta/llama-3.1-8b-instruct", "your-api-key")
    >>> response = backend("You are a helpful assistant.", "What is the capital of France?")
    >>> print(response)
    The capital of France is Paris.
    """
    return NIMBackend(model_name=model_name, api_key=api_key, api_base=api_base)
