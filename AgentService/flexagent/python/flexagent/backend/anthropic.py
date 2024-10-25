from typing import Any, Dict, List, Optional

from .base_backend import BaseBackend
from .registry import reg
from .sampling_params import SamplingParams


class AnthropicBackend(BaseBackend):
    """
    A backend class for interacting with Anthropic's language models.

    This class provides an interface to generate text using Anthropic's API.
    It inherits from BaseBackend and implements the necessary methods for text generation.

    :param model_name: The name of the Anthropic model to use.
    :param api_key: The API key for authenticating with Anthropic's API.
    :param api_base: The base URL for the Anthropic API (optional).
    """

    def __init__(self, model_name: str, api_key: str, api_base: Optional[str] = None):
        super().__init__(model_name)
        import anthropic

        self.client: anthropic.Anthropic = anthropic.Anthropic(
            api_key=api_key, base_url=api_base
        )

    def generate(self, messages: List[Dict[str, str]], *args, **kwargs) -> str:
        """
        Generate text using the Anthropic model.

        This method takes a system prompt and a user prompt, combines them into a message format
        expected by Anthropic's API, and generates a response.

        :param messages: The messages to generate text from.
        :param args: Additional positional arguments (not used in this implementation).
        :param kwargs: Additional keyword arguments. Can include 'sampling_params'.
        :return: The generated text response from the model.

        :example:

        >>> backend = AnthropicBackend("claude-2", "your-api-key")
        >>> system_prompt = "You are a helpful assistant."
        >>> user_prompt = "What is the capital of France?"
        >>> response = backend.generate(system_prompt, user_prompt)
        >>> print(response)
        The capital of France is Paris.
        """

        sampling_params = kwargs.get("sampling_params", SamplingParams())
        anthropic_params = sampling_params.to_anthropic_params()

        response = self.client.messages.create(
            model=self.model_name, messages=messages, **anthropic_params
        )

        return response.content[0].text.strip()

    def __call__(self, *args: Any, **kwds: Any) -> Dict[str, Any]:
        """
        Make the AnthropicBackend instance callable.

        This method allows the instance to be called directly, which in turn calls the generate method.

        :param args: Positional arguments to pass to the generate method.
        :param kwds: Keyword arguments to pass to the generate method.
        :return: A dictionary containing the generated text.

        :example:

        >>> backend = AnthropicBackend("claude-2", "your-api-key")
        >>> result = backend("You are a helpful assistant.", "What is the capital of France?")
        >>> print(result)
        {'generated_text': 'The capital of France is Paris.'}
        """
        generated_text = self.generate(*args, **kwds)
        return {"generated_text": generated_text}


@reg("anthropic")
def get_anthropic_backend(
    model_name: str, api_key: str, api_base: Optional[str] = None
):
    """
    Factory function to create an AnthropicBackend instance.

    This function is registered with the 'anthropic' key and can be used to create
    AnthropicBackend instances dynamically.

    :param model_name: The name of the Anthropic model to use.
    :param api_key: The API key for authenticating with Anthropic's API.
    :param api_base: The base URL for the Anthropic API (optional).
    :return: An instance of AnthropicBackend.

    :example:

    >>> backend = get_anthropic_backend("claude-2", "your-api-key")
    >>> response = backend.generate("You are a helpful assistant.", "What is the capital of France?")
    >>> print(response)
    The capital of France is Paris.
    """
    return AnthropicBackend(model_name, api_key, api_base)
