# Copyright (c) 2024 NVIDIA Corporation

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BackendConfig:
    backend_type: str
    model_name: str
    api_key: str = None
    api_base: str = None


class BaseBackend(ABC):
    """
    An abstract base class for implementing backend interfaces for different language models.

    This class provides a common structure for various backend implementations,
    allowing for consistent interaction with different language models.

    Attributes:
        _model_name (str): The name of the language model being used.
        client: The client object for interacting with the language model API (if applicable).

    Example:
        class MyBackend(BaseBackend):
            def __init__(self, model_name: str):
                super().__init__(model_name)
                self.client = MyAPIClient()

            def generate(self, prompt: str) -> str:
                return self.client.generate_text(prompt)
    """

    def __init__(self, model_name: str, *args, **kwargs):
        """
        Initialize the BaseBackend.

        Args:
            model_name (str): The name of the language model to be used.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self._model_name = model_name
        self.client = None

    @property
    def model_name(self) -> str:
        """
        Get the name of the current language model.

        Returns:
            str: The name of the language model.
        """
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str):
        """
        Set the name of the language model.

        Args:
            model_name (str): The new name for the language model.
        """
        self._model_name = model_name

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], *args, **kwargs) -> str:
        """
        Generate text based on the given prompt using the language model.

        This method must be implemented by subclasses to define the specific
        text generation logic for the chosen language model.

        Args:
            prompt (str): The input prompt for text generation.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The generated text response.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Example:
            def generate(self, prompt: str, max_tokens: int = 100) -> str:
                response = self.client.complete(prompt, max_tokens=max_tokens)
                return response.text
        """
        pass
