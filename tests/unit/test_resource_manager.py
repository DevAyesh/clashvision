"""
Unit tests for ClashVision Centralized Lifecycle and Resource Manager.
Verifies Phase 9 Requirement:
- ResourceManager singleton lifecycle
- Hook registration and unregistration
- Clean LIFO execution order on cleanup_all
- Exception isolation during cleanup
"""

import unittest
from clashvision.core.resource_manager import ResourceManager, get_resource_manager


class TestResourceManager(unittest.TestCase):
    """Verifies ResourceManager singleton and deterministic cleanup semantics."""

    def setUp(self):
        self.mgr = get_resource_manager()
        self.mgr.reset()

    def tearDown(self):
        self.mgr.reset()

    def test_singleton_identity(self):
        m1 = get_resource_manager()
        m2 = ResourceManager()
        self.assertIs(m1, m2)

    def test_hook_registration_and_execution_order(self):
        execution_order = []

        def hook1():
            execution_order.append("resource_1")

        def hook2():
            execution_order.append("resource_2")

        self.mgr.register_cleanup_hook(hook1)
        self.mgr.register_cleanup_hook(hook2)

        # Cleanup should execute LIFO (hook2 then hook1)
        executed = self.mgr.cleanup_all()
        self.assertEqual(executed, 2)
        self.assertEqual(execution_order, ["resource_2", "resource_1"])

    def test_unregister_hook(self):
        calls = []

        def hook():
            calls.append(1)

        self.mgr.register_cleanup_hook(hook)
        self.mgr.unregister_cleanup_hook(hook)

        executed = self.mgr.cleanup_all()
        self.assertEqual(executed, 0)
        self.assertEqual(calls, [])

    def test_exception_isolation_during_cleanup(self):
        executed_after_crash = []

        def buggy_hook():
            raise RuntimeError("Database connection timed out during close")

        def good_hook():
            executed_after_crash.append("success")

        # Even if buggy_hook fails, good_hook must still execute
        self.mgr.register_cleanup_hook(good_hook)
        self.mgr.register_cleanup_hook(buggy_hook)

        executed = self.mgr.cleanup_all()
        self.assertEqual(executed, 1)
        self.assertEqual(executed_after_crash, ["success"])


if __name__ == "__main__":
    unittest.main()

