#!/usr/bin/env python3
"""
ClashVision Self-Contained Offline Test Runner.
Executes test suites across Tiers 1–4 and synthetic fixtures.
Supports pytest with seamless fallback to standard library unittest.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


BANNER = r"""
================================================================================
   ______ _           _      _   _ _     _             
  / ____/| |         | |    | | | (_)   (_)            
 | |     | | __ _ ___| |__  | | | |_ ___ _  ___  _ __  
 | |     | |/ _` / __| '_ \ | | | | / __| |/ _ \| '_ \ 
 | |____ | | (_| \__ \ | | | \ \_/ / \__ \ | (_) | | | |
  \_____||_|\__,_|___/_| |_|  \___/|_|___/_|\___/|_| |_|
        Automated Offline Test Suite Runner
================================================================================
"""


def run_with_pytest(args: argparse.Namespace) -> int:
    """Run tests using pytest with structured options."""
    import pytest

    pytest_args = ["-v", "--durations=10"]

    if args.tier:
        tier_map = {
            1: "tests/tiers/test_tier1_features.py",
            2: "tests/tiers/test_tier2_boundaries.py",
            3: "tests/tiers/test_tier3_pairwise.py",
            4: "tests/tiers/test_tier4_workloads.py",
        }
        target = tier_map.get(args.tier)
        if target:
            pytest_args.append(target)
        else:
            print(f"[ERROR] Invalid tier: {args.tier}. Choose 1, 2, 3, or 4.")
            return 1
    elif args.fixtures_only:
        pytest_args.append("tests/fixtures/")
    else:
        pytest_args.append("tests/")

    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    print(f"Executing: pytest {' '.join(pytest_args)}")
    return int(pytest.main(pytest_args))


def run_with_unittest(args: argparse.Namespace) -> int:
    """Fallback runner using standard library unittest."""
    import unittest

    print("[INFO] Running in fallback mode via standard library unittest...")
    loader = unittest.TestLoader()

    start_dir = "tests"
    if args.tier:
        pattern = f"test_tier{args.tier}*.py"
        start_dir = "tests/tiers"
    elif args.fixtures_only:
        pattern = "test_*.py"
        start_dir = "tests/fixtures"
    else:
        pattern = "test_*.py"

    suite = loader.discover(start_dir=start_dir, pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="ClashVision Offline Test Suite Runner")
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3, 4],
        help="Execute specific test tier (1: Features, 2: Boundaries, 3: Pairwise, 4: Workloads)",
    )
    parser.add_argument(
        "--fixtures-only",
        action="store_true",
        help="Execute only synthetic fixture generation tests",
    )
    parser.add_argument(
        "-k",
        "--keyword",
        type=str,
        default="",
        help="Filter tests by keyword expression",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Force standard library unittest runner instead of pytest",
    )

    args = parser.parse_args()

    print(BANNER)
    start_time = time.perf_counter()

    use_pytest = False
    if not args.fallback:
        try:
            import pytest
            use_pytest = True
        except ImportError:
            use_pytest = False

    if use_pytest:
        exit_code = run_with_pytest(args)
    else:
        exit_code = run_with_unittest(args)

    duration = time.perf_counter() - start_time
    status = "SUCCESS" if exit_code == 0 else "FAILURE"
    print("\n" + "=" * 80)
    print(f"  Test Execution Result: {status} (Exit Code: {exit_code})")
    print(f"  Total Duration: {duration:.2f} seconds")
    print("=" * 80 + "\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
