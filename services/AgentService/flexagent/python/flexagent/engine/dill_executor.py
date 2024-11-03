import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Callable, Optional, Tuple

import dill


class DillProcessPoolExecutor(ProcessPoolExecutor):
    """
    A subclass of ProcessPoolExecutor that uses dill for serialization.

    This class extends the functionality of ProcessPoolExecutor by using dill
    to serialize and deserialize functions and arguments. This allows for
    more complex objects and functions to be passed to the executor.

    :param max_workers: The maximum number of processes to use for execution.
                        If None, it will default to the number of processors on the machine.
    :param mp_context: A multiprocessing context to use for multiprocessing.
    :param initializer: An optional callable used to initialize worker processes.
    :param initargs: Arguments to pass to the initializer.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        mp_context: Optional[multiprocessing.context.BaseContext] = None,
        initializer: Optional[Callable[..., None]] = None,
        initargs: Tuple[Any, ...] = (),
    ) -> None:
        super().__init__(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=initializer,
            initargs=initargs,
        )

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """
        Submit a callable to be executed with the given arguments.

        :param fn: The callable to be executed.
        :param args: Positional arguments for the callable.
        :param kwargs: Keyword arguments for the callable.
        :return: A Future representing the execution of the callable.
        """
        # Serialize the function and arguments using dill
        fn_pkl = dill.dumps(fn)
        args_pkl = dill.dumps(args)
        kwargs_pkl = dill.dumps(kwargs)
        # Submit the wrapper function to the executor
        return super().submit(self._dill_wrapper, fn_pkl, args_pkl, kwargs_pkl)

    @staticmethod
    def _dill_wrapper(fn_pkl: bytes, args_pkl: bytes, kwargs_pkl: bytes) -> Any:
        """
        Internal method to deserialize and execute the function.

        :param fn_pkl: Pickled function.
        :param args_pkl: Pickled positional arguments.
        :param kwargs_pkl: Pickled keyword arguments.
        :return: The result of the function execution.
        """
        # Deserialize the function and arguments
        fn = dill.loads(fn_pkl)
        args = dill.loads(args_pkl)
        kwargs = dill.loads(kwargs_pkl)
        # Execute the function
        return fn(*args, **kwargs)
