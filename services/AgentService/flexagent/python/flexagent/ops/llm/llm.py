# # Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Any, Dict, List, Optional
import jinja2
import json
from flexagent.backend import BackendConfig, get
from flexagent.engine import Operator, Resource, Value


def compute(messages: List[Dict[str, str]], **kwargs):
    config = kwargs.pop("backend")
    backend_config = {
        k: v
        for k, v in config.__dict__.items()
        if k != "backend_type" and v is not None
    }
    backend = get(config.backend_type)(**backend_config)
    return backend.generate(messages, **kwargs)

TOOL_USE_TEMPLATE = jinja2.Template("""
You have the following tools available to you: [{{tool_names}}]

                                    
{{tool_schemas}}

When you think you should use a tool, you should generate a JSON object following the schema of the tool.
```tool
{
    "name": "tool_name",
    "parameters": {
        "param_name": "param_value",
        ...
    }
}
```
If you don't have any tool to use, just answer the question and let the user know that you don't have any tool to use.                      
""")

class LLM(Operator):
    """
    A class representing a Language Model (LLM) operator.

    This class extends the Operator class and provides functionality to interact
    with various LLM backends.

    :param backend: An optional BackendConfig object specifying the LLM backend configuration.
    :type backend: Optional[BackendConfig]

    :example:
    >>> from flexagent.backend import BackendConfig
    >>> backend_config = BackendConfig(backend_type="openai", api_key="your-api-key")
    >>> llm = LLM(backend=backend_config)
    """

    def __init__(self, backend: Optional[BackendConfig] = None):
        super().__init__(compute)
        self.backend = backend
        self.tools = []

    def to(self, backend: BackendConfig):
        """
        Set or change the LLM backend configuration.

        :param backend: A BackendConfig object specifying the new LLM backend configuration.
        :type backend: BackendConfig
        :return: The LLM instance with updated backend configuration.
        :rtype: LLM
        :raises ValueError: If the provided backend is not a subclass of BaseBackend.

        :example:
        >>> new_backend_config = BackendConfig(backend_type="anthropic", api_key="your-api-key")
        >>> llm.to(new_backend_config)
        """
        if not isinstance(backend, BackendConfig):
            raise ValueError("Backend must be a subclass of BaseBackend")
        self.backend = backend
        return self

    def __call__(
        self, *args: Any, resources: Optional[List[Resource]] = None, **kwargs: Any
    ) -> Value:
        """
        Call the LLM operator with the specified arguments and resources.

        :param args: Positional arguments to be passed to the compute function.
        :param resources: Optional list of Resource objects.
        :type resources: Optional[List[Resource]]
        :param kwargs: Keyword arguments to be passed to the compute function.
        :return: The result of the LLM computation.
        :rtype: Value
        :raises ValueError: If the LLM backend is not set.

        :example:
        >>> result = llm("Hello, how are you?", max_tokens=50)
        >>> print(result)
        """
        # Include backend in kwargs
        kwargs["backend"] = self.backend
        if self.backend is None:
            raise ValueError("LLM backend is not set")
        if len(self.tools) > 0:
            kwargs["tools"] = self.tools
        return super().__call__(*args, resources=resources, **kwargs)

    @staticmethod
    def get_function_schema() -> Dict[str, Any]:
        return {}

    def enable_tools(self, tools: List[Operator]):
        tool_schemas = []
        tool_names = []
        for tool in tools:
            tool_schemas.append(tool.get_function_schema())
            tool_names.append(json.loads(tool.get_function_schema())["name"])
        tool_schemas = "\n".join(tool_schemas)
        tool_names = ", ".join(tool_names)
        self.tools = TOOL_USE_TEMPLATE.render(tool_schemas=tool_schemas, tool_names=tool_names)

