# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .engine import get_engine, Resource
from .value import Value


class Operator(ABC):
    """
    A class that wraps a function to be executed as an operation in the engine.

    The Operator class is responsible for submitting tasks to the engine and
    managing the input and output values for the wrapped function.

    :param func: The function to be wrapped and executed as an operation.
    :type func: Callable[..., Any]

    Attributes:
        engine: The engine instance used for task submission.
        func: The wrapped function to be executed.

    Warning:
        The wrapped function (func) must be picklable. Non-picklable functions
        may cause deadlocks when submitted to the engine.

    Example:
        >>> def add(a, b):
        ...     return a + b
        >>> op = Operator(add)
        >>> result = op(3, 4)
        >>> print(result)  # This will print a Value object
    """

    def __init__(self, func: Callable[..., Any]):
        self.engine = get_engine()
        self.func = func

    @staticmethod
    @abstractmethod
    def get_function_schema(cls) -> Dict[str, Any]:
        pass

    def __call__(
        self, *args: Any, resources: Optional[List[Resource]] = None, **kwargs: Any
    ) -> Value:
        """
        Execute the wrapped function as an operation in the engine.

        This method prepares the input values, submits the task to the engine,
        and returns a Value object representing the output.

        :param args: Positional arguments to be passed to the wrapped function.
        :type args: Any
        :param resources: Optional list of resources required for the operation.
        :type resources: Optional[List[Resource]]
        :param kwargs: Keyword arguments to be passed to the wrapped function.
        :type kwargs: Any
        :return: A Value object representing the output of the operation.
        :rtype: Value

        :raises: Any exception that may be raised by the wrapped function or the engine.

        Note:
            - If an argument is a Value object, it will be treated as both an input value
              and a function argument.
            - All other arguments are passed directly to the wrapped function.
            - The method returns immediately with a Value object, which will be populated
              with the result once the task is completed by the engine.
        """
        input_values: List[Value] = []
        func_args: List[Any] = []
        for arg in args:
            if isinstance(arg, Value):
                input_values.append(arg)
                func_args.append(arg)
            else:
                func_args.append(arg)
        out = Value()
        output_values = [out]
        self.engine.submit_task(
            self.func, func_args, kwargs, input_values, output_values, resources
        )
        return out
