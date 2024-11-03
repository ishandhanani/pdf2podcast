# Copyright (c) 2024 NVIDIA Corporation

from typing import Callable, Optional


BACKEND_REGISTRY: dict[str, Callable[[], None]] = {}


def reg(func_name: str, func: Optional[Callable[[], None]] = None) -> Callable:
    """
    Register a backend function in the BACKEND_REGISTRY.

    This decorator can be used in two ways:
    1. As a decorator without arguments: @reg("func_name")
    2. As a function call: reg("func_name", func)

    :param func_name: A string identifier for the backend function
    :param func: The function to be registered (optional)
    :return: The registered function
    :raises ValueError: If a backend with the same name is already registered

    Examples:
    ---------
    >>> @reg("my_backend")
    ... def my_backend_func():
    ...     pass

    >>> def another_backend():
    ...     pass
    >>> reg("another_backend", another_backend)
    """
    if func_name in BACKEND_REGISTRY:
        raise ValueError(f"Backend {func_name} already registered")

    def _do_reg(func: Callable[[], None]) -> Callable[[], None]:
        BACKEND_REGISTRY[func_name] = func
        return func

    return _do_reg if func is None else _do_reg(func)


def get(func_name: str) -> Callable:
    """
    Retrieve a registered backend function from the BACKEND_REGISTRY.

    :param func_name: The string identifier of the backend function to retrieve
    :return: The registered backend function
    :raises ValueError: If the requested backend is not found in the registry

    Example:
    --------
    >>> registered_func = get("my_backend")
    >>> registered_func()
    """
    try:
        return BACKEND_REGISTRY[func_name]
    except KeyError:
        raise ValueError(f"Backend {func_name} not registered")
