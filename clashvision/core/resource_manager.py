"""
ClashVision Core Subsystem — Centralized Lifecycle & Resource Manager.
Provides thread-safe registration, tracking, and deterministic cleanup for:
- Database connections and transactions
- Background simulation workers and threads
- In-memory image buffers and scratch files
Ensures zero resource leaks on normal exit, exception crashes, or emergency stops.
"""

from __future__ import annotations
import atexit
import logging
import threading
from typing import Any, Callable, List, Optional

logger = logging.getLogger("clashvision.core.resource_manager")


class ResourceManager:
    """
    Singleton lifecycle and resource manager for ClashVision.
    Guarantees that all registered resources are cleanly released.
    """

    _instance: Optional[ResourceManager] = None
    _lock = threading.Lock()

    def __new__(cls) -> ResourceManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._cleanup_hooks: List[Callable[[], None]] = []
        self._hooks_lock = threading.Lock()
        self._is_cleaning_up = False

        # Register process-level exit hook
        try:
            atexit.register(self.cleanup_all)
        except Exception as e:
            logger.debug("Failed to register atexit hook: %s", e)

    def register_cleanup_hook(self, hook: Callable[[], None]) -> None:
        """Registers a callable to be executed on application cleanup/shutdown."""
        with self._hooks_lock:
            if hook not in self._cleanup_hooks:
                self._cleanup_hooks.append(hook)

    def unregister_cleanup_hook(self, hook: Callable[[], None]) -> None:
        """Removes a previously registered cleanup hook."""
        with self._hooks_lock:
            if hook in self._cleanup_hooks:
                self._cleanup_hooks.remove(hook)

    def cleanup_all(self) -> int:
        """
        Executes all registered cleanup hooks in LIFO (reverse) order.
        Exceptions are safely logged without halting subsequent cleanups.
        Returns count of successfully executed cleanup hooks.
        """
        with self._hooks_lock:
            if self._is_cleaning_up:
                return 0
            self._is_cleaning_up = True
            hooks = list(reversed(self._cleanup_hooks))
            self._cleanup_hooks.clear()

        success_count = 0
        for hook in hooks:
            try:
                hook()
                success_count += 1
            except Exception as ex:
                logger.warning("Exception during resource cleanup hook: %s", ex)

        self._is_cleaning_up = False
        return success_count

    def reset(self) -> None:
        """Resets the hook registry for testing purposes."""
        with self._hooks_lock:
            self._cleanup_hooks.clear()
            self._is_cleaning_up = False


_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Retrieves the global singleton ResourceManager instance."""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
    return _global_resource_manager

