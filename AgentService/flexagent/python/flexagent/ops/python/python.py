# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import subprocess
import sys
import tempfile

from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from flexagent.engine import (
    Operator,
    OperatorInputsSchema,
    OperatorSchema,
    Resource,
    Value,
)


def compute(code: str, **kwargs: Any):
    """
    Execute a Python code snippet in a temporary file and return the results.

    This function creates a temporary file with the given Python code, executes it
    using the current Python interpreter, captures the output, and then removes the
    temporary file.

    Parameters
    ----------
    code : str
        The Python code to be executed.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'stdout': str, the standard output of the executed code.
        - 'stderr': str, the standard error output of the executed code.
        - 'exit_code': int, the exit code of the process.

    Examples
    --------
    >>> result = compute("print('Hello, World!')")
    >>> print(result['stdout'])
    Hello, World!

    >>> result = compute("import sys; print(sys.version)")
    >>> print(result['stdout'])
    3.8.10 (default, May 26 2023, 14:05:08)
    [GCC 9.4.0]

    >>> result = compute("1/0")
    >>> print(result['stderr'])
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ZeroDivisionError: division by zero
    """
    timeout = kwargs.pop("timeout", 360)
    # create a temp file to write the program
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(code.encode("utf-8"))
        temp_file_path = temp_file.name
    # run the program
    process = subprocess.Popen(
        [sys.executable, temp_file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        stderr = "Timeout error: The code execution exceeded the time limit.".encode(
            "utf-8"
        )
    # check the exit code
    exit_code = process.returncode
    # remove the temp file
    os.remove(temp_file_path)
    return {
        "stdout": stdout.decode("utf-8"),
        "stderr": stderr.decode("utf-8"),
        "exit_code": exit_code,
    }


class PythonOpInputSchema(OperatorInputsSchema):
    code: str = Field(..., description="The Python code to be executed.")


class PythonOpSchema(OperatorSchema):
    name: ClassVar[str] = "python"
    description: ClassVar[str] = (
        "Execute a Python code snippet in a temporary file and return the results."
    )
    parameters: ClassVar[Type[BaseModel]] = PythonOpInputSchema


class Python(Operator):
    """
    An Operator class for executing Python code.

    This class extends the Operator class to provide functionality for
    executing Python code snippets.

    Parameters
    ----------
    resource_dependencies : Optional[List[Resource]], optional
        A list of Resource objects that this operator depends on.

    Examples
    --------
    >>> python_op = Python()
    >>> result = python_op("print('Hello from Python Operator!')")
    >>> print(result['stdout'])
    Hello from Python Operator!

    >>> python_op_with_deps = Python(resource_dependencies=[some_resource])
    >>> result = python_op_with_deps("import some_module; print(some_module.version)")
    >>> print(result['stdout'])
    1.2.3
    """

    def __init__(self, timeout: int = 360):
        super().__init__(compute)
        self.timeout = timeout

    def __call__(
        self, *args: Any, resources: Optional[List[Resource]] = None, **kwargs: Any
    ) -> Value:
        return super().__call__(
            *args, timeout=self.timeout, resources=resources, **kwargs
        )

    @staticmethod
    def get_function_schema() -> Dict[str, Any]:
        return PythonOpSchema.get_function_schema()
