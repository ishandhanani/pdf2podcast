# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import multiprocessing
import os
import threading
from collections import defaultdict, deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from .dill_executor import DillProcessPoolExecutor

from .value import Value


@dataclass(frozen=True)
class Resource:
    """
    Represents a resource that can be used by tasks in the dependency engine.

    :param name: The name of the resource.
    :type name: str
    """

    name: str


class DepEngine:
    """
    A dependency engine that manages task execution with resource constraints.

    This class handles task submission, dependency tracking, and resource management
    for concurrent task execution using a process pool.

    Examples:
        >>> engine = DepEngine()
        >>> def task_function(x, y):
        ...     return x + y
        >>> input_value = Value()
        >>> output_value = Value()
        >>> engine.submit_task(task_function, (5, 3), {}, [input_value], [output_value])
        >>> engine.shutdown()
    """

    def __init__(self) -> None:
        """
        Initialize the DepEngine with a process pool executor and necessary data structures.

        The engine uses a ProcessPoolExecutor with a number of workers equal to the number
        of CPU cores available on the system.

        Examples:
            >>> engine = DepEngine()
            >>> isinstance(engine.executor, ProcessPoolExecutor)
            True
        """
        max_workers = int(os.environ.get("MAX_WORKERS", os.cpu_count()))
        # self.executor: ProcessPoolExecutor = ProcessPoolExecutor(
        #     max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")
        # )
        self.executor: DillProcessPoolExecutor = DillProcessPoolExecutor(
            max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")
        )
        self.tasks: Dict[int, Dict[str, Any]] = {}
        self.task_counter: int = 0
        self.lock: threading.RLock = threading.RLock()
        self.active_resources: Set[Resource] = set()
        self.resource_queues: Dict[Resource, Deque[int]] = defaultdict(deque)

    def submit_task(
        self,
        func: Callable,
        args: Tuple,
        kwargs: Dict[str, Any],
        input_values: List[Value],
        output_values: List[Value],
        resources: Optional[Set[Resource]] = None,
    ) -> None:
        """
        Submit a task to the dependency engine for execution.

        This method adds the task to the engine's queue, considering its dependencies
        and resource requirements. If all dependencies are met and required resources
        are available, the task is immediately submitted for execution.

        Args:
            func (Callable): The function to be executed.
            args (Tuple): Positional arguments for the function.
            kwargs (Dict[str, Any]): Keyword arguments for the function.
            input_values (List[Value]): List of input Value objects.
            output_values (List[Value]): List of output Value objects.
            resources (Optional[Set[Resource]]): Set of Resource objects required by the task.

        Examples:
            >>> engine = DepEngine()
            >>> def add(x, y):
            ...     return x + y
            >>> input_val1, input_val2 = Value(), Value()
            >>> output_val = Value()
            >>> input_val1.data, input_val2.data = 5, 3
            >>> input_val1.event.set(), input_val2.event.set()
            >>> engine.submit_task(add, (input_val1, input_val2), {}, [input_val1, input_val2], [output_val])
            >>> # The task will be executed and the result will be stored in output_val
        """
        with self.lock:
            task_id = self.task_counter
            self.task_counter += 1
            task = {
                "func": func,
                "args": args,
                "kwargs": kwargs,
                "input_values": input_values,
                "output_values": output_values,
                "dependencies": set(),
                "resources": resources or set(),
            }
            self.tasks[task_id] = task  # Store the task
            for val in input_values:
                if not val.event.is_set():
                    task["dependencies"].add(val)
                    val.dependents.add(task_id)
            if not task["dependencies"]:
                # All inputs are ready, try to submit the task
                self._try_submit_task(task_id, task)

    def _try_submit_task(self, task_id: int, task: Dict[str, Any]) -> None:
        """
        Attempt to submit a task for execution if resources are available.

        This internal method checks if the required resources for the task are available.
        If so, it marks the resources as in use and submits the task for execution.
        Otherwise, it enqueues the task for later execution.

        Args:
            task_id (int): The unique identifier of the task.
            task (Dict[str, Any]): The task dictionary containing all necessary information.

        Note:
            This method assumes that the engine's lock is held when it's called.
        """
        # Assumes lock is held
        if self._resources_available(task["resources"]):
            # Mark resources as in use
            self.active_resources.update(task["resources"])
            # Proceed to submit the task
            func = task["func"]
            args = []
            for arg in task["args"]:
                if isinstance(arg, Value):
                    args.append(arg.data)
                else:
                    args.append(arg)
            future = self.executor.submit(func, *args, **task["kwargs"])
            future.task_id = task_id
            future.add_done_callback(self._task_done)
        else:
            # Resources are not available, enqueue the task per resource
            for resource in task["resources"]:
                self.resource_queues[resource].append(task_id)

    def _resources_available(self, resources: Set[Resource]) -> bool:
        """
        Check if all required resources are available.

        This method determines whether the set of required resources for a task
        is disjoint from the set of currently active (in-use) resources.

        Args:
            resources (Set[Resource]): Set of resources to check.

        Returns:
            bool: True if all resources are available, False otherwise.

        Examples:
            >>> engine = DepEngine()
            >>> res1, res2 = Resource("CPU"), Resource("GPU")
            >>> engine.active_resources = {res1}
            >>> engine._resources_available({res2})
            True
            >>> engine._resources_available({res1, res2})
            False
        """
        return resources.isdisjoint(self.active_resources)

    def _task_done(self, future: Future) -> None:
        """
        Callback function to handle task completion.

        This method is called when a task completes execution. It processes the task's
        result, updates the output values, releases resources, and checks if any
        waiting tasks can now proceed.

        Args:
            future (Future): The completed Future object.

        Note:
            This method handles both successful task completions and exceptions.
        """
        task_id = future.task_id  # type: ignore
        try:
            result = future.result()
        except Exception as e:
            result = e
        with self.lock:
            task = self.tasks.pop(task_id, None)
            if task is None:
                return
            # Release resources
            self.active_resources.difference_update(task["resources"])
            # Check if any waiting tasks can now proceed
            self._release_waiting_tasks(task["resources"])
            if isinstance(result, Exception):
                # Handle exception
                result = (result,)
            else:
                if not isinstance(result, tuple):
                    result = (result,)
            for val, res in zip(task["output_values"], result):
                val.data = res
                val.event.set()
                # Notify dependents
                for dep_task_id in val.dependents.copy():
                    dep_task = self.tasks.get(dep_task_id)
                    if dep_task:
                        dep_task["dependencies"].remove(val)
                        if not dep_task["dependencies"]:
                            # All dependencies are resolved, try to submit the task
                            self._try_submit_task(dep_task_id, dep_task)

    def _release_waiting_tasks(self, released_resources: Set[Resource]) -> None:
        """
        Attempt to submit waiting tasks after resources have been released.

        This method checks if any tasks waiting for the released resources can now
        be submitted for execution.

        Args:
            released_resources (Set[Resource]): Set of resources that have been released.

        Note:
            This method is called internally after a task completes and releases its resources.
        """
        # Try to submit tasks waiting for the released resources
        for resource in released_resources:
            queue = self.resource_queues[resource]
            tasks_to_try = []
            while queue:
                task_id = queue.popleft()
                task = self.tasks.get(task_id)
                if (
                    task
                    and self._resources_available(task["resources"])
                    and not task["dependencies"]
                ):
                    tasks_to_try.append((task_id, task))
                else:
                    # If resources are still not available or task has dependencies, re-enqueue
                    queue.appendleft(task_id)
                    break  # Exit to prevent infinite loop
            for task_id, task in tasks_to_try:
                self.active_resources.update(task["resources"])
                func = task["func"]
                args = []
                for arg in task["args"]:
                    if isinstance(arg, Value):
                        args.append(arg.data)
                    else:
                        args.append(arg)
                future = self.executor.submit(func, *args, **task["kwargs"])
                future.task_id = task_id
                future.add_done_callback(self._task_done)

    def shutdown(self) -> None:
        """
        Shutdown the dependency engine and its executor.

        This method waits for all pending tasks to complete before shutting down
        the ProcessPoolExecutor.

        Examples:
            >>> engine = DepEngine()
            >>> # ... submit some tasks ...
            >>> engine.shutdown()
            >>> # All tasks are completed and resources are released
        """
        self.executor.shutdown(wait=True)


# Singleton engine instance
_engine: Optional[DepEngine] = None
_engine_lock: threading.Lock = threading.Lock()
_engine_thread: Optional[threading.Thread] = None


def initialize_engine() -> DepEngine:
    """
    Initialize and return the singleton DepEngine instance.

    If called from a non-main process, it starts the engine in a new thread.

    :return: The initialized DepEngine instance.
    :rtype: DepEngine
    """
    global _engine
    global _engine_thread
    with _engine_lock:
        if _engine is None:
            if multiprocessing.current_process().name == "MainProcess":
                _engine = DepEngine()
            else:
                # pop out warning
                print("__main__ is not detected, starting the engine in a new thread")
                assert _engine_thread is None, "Engine thread already exists"
                _engine_thread = threading.Thread(target=initialize_engine)
                _engine_thread.start()

    return _engine


def get_engine() -> DepEngine:
    """
    Get the singleton DepEngine instance, initializing it if necessary.

    :return: The DepEngine instance.
    :rtype: DepEngine
    """
    global _engine
    if _engine is None:
        initialize_engine()
    with _engine_lock:
        return _engine


def shutdown_engine() -> None:
    """
    Shutdown the singleton DepEngine instance and join any associated threads.
    """
    global _engine
    global _engine_thread
    with _engine_lock:
        if _engine is not None:
            _engine.shutdown()
            _engine = None
        if _engine_thread is not None:
            _engine_thread.join()
            _engine_thread = None
