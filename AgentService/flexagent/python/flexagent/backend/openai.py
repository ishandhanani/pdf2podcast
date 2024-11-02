from typing import Any, Dict, List, Optional

from .base_backend import BaseBackend
from .registry import reg
from .sampling_params import SamplingParams


class OpenAIBackend(BaseBackend):
    """
    A backend class for interacting with OpenAI's API.

    This class provides methods to generate text using OpenAI's language models.

    :param model_name: The name of the OpenAI model to use.
    :param api_key: The API key for authentication with OpenAI.
    :param org_id: Optional organization ID for OpenAI API access.
    :param api_base: Optional custom API base URL.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        org_id: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        super().__init__(model_name)
        import openai

        self.client = openai.OpenAI(
            api_key=api_key, organization=org_id, base_url=api_base
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[SamplingParams] = None,
        *args: Any,
        **kwargs: Any
    ) -> str:
        """
        Generate text using the OpenAI API.

        :param messages: The list of messages to generate text from.
        :param sampling_params: Optional sampling parameters to control text generation.
        :param args: Additional positional arguments to pass to the API call.
        :param kwargs: Additional keyword arguments to pass to the API call.
        :return: The generated text response.

        :example:

        >>> backend = OpenAIBackend("gpt-3.5-turbo", "your-api-key")
        >>> response = backend.generate(
        ...     messages=[
        ...         {"role": "system", "content": "You are a helpful assistant."},
        ...         {"role": "user", "content": "What is the capital of France?"}
        ...     ],
        ...     sampling_params=SamplingParams(temperature=0.7, max_new_tokens=50)
        ... )
        >>> print(response)
        The capital of France is Paris.
        """

        if sampling_params:
            kwargs.update(
                {
                    "temperature": sampling_params.temperature,
                    "max_tokens": sampling_params.max_new_tokens,
                    "top_p": sampling_params.top_p,
                    "frequency_penalty": sampling_params.frequency_penalty,
                    "presence_penalty": sampling_params.presence_penalty,
                }
            )

        if "tools" in kwargs:
            tool_msg = {
                "role": "system",
                "content": kwargs["tools"]
            }
            messages = [tool_msg, *messages]
        print(f"{messages=}")
        response = self.client.chat.completions.create(
            model=self.model_name, stream=True, messages=messages, *args, **kwargs
        )

        accumulated_content = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                accumulated_content += chunk.choices[0].delta.content

        return accumulated_content

    def __call__(self, *args: Any, **kwds: Any) -> str:
        """
        Allow the class instance to be called as a function.

        This method simply calls the `generate` method with the provided arguments.

        :param args: Positional arguments to pass to the `generate` method.
        :param kwds: Keyword arguments to pass to the `generate` method.
        :return: The generated text response.
        """
        return self.generate(*args, **kwds)


@reg("openai")
def get_openai_backend(
    model_name: str,
    api_key: str,
    org_id: Optional[str] = None,
    api_base: Optional[str] = None,
) -> OpenAIBackend:
    """
    Factory function to create an OpenAIBackend instance.

    This function is registered with the "openai" key and can be used to create
    OpenAIBackend instances.

    :param model_name: The name of the OpenAI model to use.
    :param api_key: The API key for authentication with OpenAI.
    :param org_id: Optional organization ID for OpenAI API access.
    :param api_base: Optional custom API base URL.
    :return: An instance of OpenAIBackend.

    :example:

    >>> backend = get_openai_backend("gpt-3.5-turbo", "your-api-key")
    >>> response = backend("You are a helpful assistant.", "What is the capital of France?")
    >>> print(response)
    The capital of France is Paris.
    """
    return OpenAIBackend(model_name, api_key, org_id, api_base)
