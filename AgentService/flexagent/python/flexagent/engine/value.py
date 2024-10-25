# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import threading
from typing import Any, Set


class Value:
    """
    A thread-safe container for a value that can be set asynchronously.

    This class is designed to hold a value that may not be immediately available,
    allowing other parts of the code to wait for the value to be set. It also
    supports dependency tracking and exception handling.

    Attributes:
        data (Any): The stored value or exception.
        event (threading.Event): An event that is set when the value is available.
        dependents (Set[Any]): A set of objects that depend on this value.

    Args:
        data (Any, optional): The initial value to store. If provided, the event
            will be set immediately. Defaults to None.

    Example:
        >>> v = Value()
        >>> v.set(42)
        >>> print(v.get())
        42
    """

    def __init__(self, data: Any = None) -> None:
        self.data: Any = data
        self.event: threading.Event = threading.Event()
        self.dependents: Set[Any] = set()
        if data is not None:
            self.event.set()

    def get(self) -> Any:
        """
        Retrieve the stored value.

        This method blocks until the value is available. If the stored value
        is an exception, it will be raised instead of returned.

        Returns:
            Any: The stored value.

        Raises:
            Exception: If the stored value is an exception, it will be raised.

        Example:
            >>> v = Value()
            >>> v.set(10)
            >>> v.get()
            10
        """
        self.event.wait()
        if isinstance(self.data, Exception):
            raise self.data
        return self.data

    def __str__(self) -> str:
        """
        Return a string representation of the stored value.

        This method automatically calls get() to retrieve the value.

        Returns:
            str: String representation of the stored value.

        Raises:
            Exception: If the stored value is an exception, it will be raised.
        """
        return str(self.get())
