# Copyright (c) 2024 NVIDIA Corporation

"""
FlexAgent: A flexible agent framework for AI applications.

This package provides tools and utilities for building and deploying
AI agents with customizable behaviors and capabilities.
"""

__version__ = "0.1.0"
__author__ = "NVIDIA Corporation"
__license__ = "Proprietary"

# Import main components
import atexit

from . import engine, ops, utils

# Convenience imports
# from .utils import logger

# executor.TaskQueue.start_engine()
atexit.register(engine.shutdown_engine)

# Define what's available when using `from flexagent import *`
__all__ = ["engine", "ops", "utils"]
