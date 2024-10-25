import dataclasses
from typing import Any, Dict, List, Union


@dataclasses.dataclass
class SamplingParams:
    """
    A dataclass representing sampling parameters for language models.

    This class encapsulates various parameters used in text generation tasks,
    such as temperature, top-p sampling, and maximum token limits.

    :param max_new_tokens: Maximum number of new tokens to generate.
    :param temperature: Controls randomness in generation. Higher values increase randomness.
    :param top_p: Cumulative probability for top-p sampling.
    :param top_k: Number of highest probability vocabulary tokens to keep for top-k sampling.
    :param repetition_penalty: Penalty for repeating tokens.
    :param stop: Token(s) at which to stop generation.
    :param frequency_penalty: Penalty for frequently used tokens.
    :param presence_penalty: Penalty for new tokens.

    :type max_new_tokens: int
    :type temperature: float
    :type top_p: float
    :type top_k: int
    :type repetition_penalty: float
    :type stop: Union[str, List[str]]
    :type frequency_penalty: float
    :type presence_penalty: float

    Example:
        >>> params = SamplingParams(max_new_tokens=100, temperature=0.8, stop=["END"])
        >>> print(params.temperature)
        0.8
    """

    max_new_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop: Union[str, List[str]] = ()
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def to_openai_params(self) -> Dict[str, Any]:
        """
        Convert sampling parameters to OpenAI API format.

        :return: A dictionary of parameters compatible with OpenAI API.
        :rtype: Dict[str, Any]

        Example:
            >>> params = SamplingParams(max_new_tokens=100, temperature=0.8, stop="END")
            >>> openai_params = params.to_openai_params()
            >>> print(openai_params)
            {'max_tokens': 100, 'temperature': 0.8, 'top_p': 0.9, 'repetition_penalty': 1.1, 'stop': 'END'}
        """
        return {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "stop": self.stop or None,
        }

    def to_anthropic_params(self) -> Dict[str, Any]:
        """
        Convert sampling parameters to Anthropic API format.

        :return: A dictionary of parameters compatible with Anthropic API.
        :rtype: Dict[str, Any]

        Example:
            >>> params = SamplingParams(max_new_tokens=100, temperature=0.8, stop=["END"])
            >>> anthropic_params = params.to_anthropic_params()
            >>> print(anthropic_params)
            {'max_tokens': 100, 'temperature': 0.8, 'top_p': 0.9, 'top_k': 50, 'stop_sequences': ['END']}
        """
        return {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop_sequences": (
                self.stop if isinstance(self.stop, (list, tuple)) else [self.stop]
            ),
        }
