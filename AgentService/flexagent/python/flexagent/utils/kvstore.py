# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Module for a thread-safe key-value store backed by SQLite.
"""

import atexit
import queue
import sqlite3
import threading
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Optional


class ChangeType(Enum):
    """Enumeration of possible change types in the key-value store."""

    INSERT = auto()
    DELETE = auto()
    UPDATE = auto()


@dataclass
class Change:
    """
    Represents a change in the key-value store.

    Attributes:
        change_type (ChangeType): The type of change (INSERT, DELETE, UPDATE).
        key (str): The key affected by the change.
        value (Optional[Any]): The new value associated with the key, if applicable.
    """

    change_type: ChangeType
    key: str
    value: Optional[Any] = None  # None for delete


class KVStore:
    """
    Singleton class for a thread-safe key-value store backed by SQLite.

    This class provides methods to set, get, and delete key-value pairs.
    Changes are queued and processed by a background thread to synchronize
    with the SQLite database.

    Attributes:
        store (dict): In-memory store for key-value pairs.
        lock (threading.Lock): Lock to ensure thread-safe operations on the store.
        change_queue (queue.Queue): Queue to hold pending changes to be written to the database.
        db_path (str): Path to the SQLite database file.
        stop_event (threading.Event): Event to signal the worker thread to stop.
        worker_thread (threading.Thread): Background thread that processes the change queue.
    """

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, db_path: str, singleton=True):
        if not singleton:
            instance = super().__new__(cls)
            instance._initialized = False
            return instance

        with cls._lock:
            if db_path not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[db_path] = instance
            return cls._instances[db_path]

    def __init__(self, db_path: str, singleton=True):
        if singleton and self._initialized:
            return
        self.store = {}
        self.lock = threading.Lock()
        self.change_queue = queue.Queue()
        self.db_path = db_path
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._initialize_db()
        self._load_from_db()
        self.worker_thread.start()
        atexit.register(self.close)
        self._initialized = True

    def _initialize_db(self):
        """
        Initialize the SQLite database and create the kv_store table if it doesn't exist.

        This method sets up the necessary table structure in the database to store key-value pairs.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value BLOB
            )
        """
        )
        conn.commit()
        conn.close()

    def _load_from_db(self):
        """
        Load existing key-value pairs from SQLite into the in-memory dictionary.

        This method reads all key-value pairs from the database and populates the in-memory store.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM kv_store")
        rows = cursor.fetchall()
        with self.lock:
            for key, value in rows:
                self.store[key] = value
        conn.close()

    def set(self, key: str, value: Any):
        """
        Set the value for a given key.

        If the key already exists, its value is updated; otherwise, a new key-value pair is inserted.

        Args:
            key (str): The key to set.
            value (Any): The value to associate with the key.
        """
        with self.lock:
            if key in self.store:
                change_type = ChangeType.UPDATE
            else:
                change_type = ChangeType.INSERT
            self.store[key] = value
            change = Change(change_type=change_type, key=key, value=value)
            self.change_queue.put(change)

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value for a given key.

        Args:
            key (str): The key to retrieve.

        Returns:
            Optional[Any]: The value associated with the key, or None if the key does not exist.
        """
        with self.lock:
            return self.store.get(key, None)

    def delete(self, key: str):
        """
        Delete a key-value pair.

        If the key exists, it is removed from the store and queued for deletion from the database.

        Args:
            key (str): The key to delete.
        """
        with self.lock:
            if key in self.store:
                del self.store[key]
                change = Change(change_type=ChangeType.DELETE, key=key)
                self.change_queue.put(change)

    def _worker(self):
        """
        Background thread that processes changes and syncs them to SQLite.

        This method runs in a separate thread, continuously processing changes from the queue
        and applying them to the database. It ensures that all changes are persisted.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        while not self.stop_event.is_set() or not self.change_queue.empty():
            try:
                # Wait for a change for a short timeout to allow graceful shutdown
                change = self.change_queue.get(timeout=0.5)
                if change.change_type == ChangeType.INSERT:
                    cursor.execute(
                        """
                        INSERT INTO kv_store (key, value) VALUES (?, ?)
                    """,
                        (change.key, change.value),
                    )
                elif change.change_type == ChangeType.UPDATE:
                    cursor.execute(
                        """
                        UPDATE kv_store SET value = ? WHERE key = ?
                    """,
                        (change.value, change.key),
                    )
                elif change.change_type == ChangeType.DELETE:
                    cursor.execute(
                        """
                        DELETE FROM kv_store WHERE key = ?
                    """,
                        (change.key,),
                    )
                conn.commit()
                self.change_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing change {change}: {e}")
        conn.close()

    def flush(self):
        """
        Wait for all pending changes to be written to SQLite.

        This method blocks until all changes in the queue have been processed and persisted.
        """
        self.change_queue.join()

    def close(self):
        """
        Stop the background worker and flush all changes.

        This method signals the worker thread to stop, waits for it to finish,
        and ensures all pending changes are written to the database.
        """
        self.stop_event.set()
        self.worker_thread.join()
        self.flush()
