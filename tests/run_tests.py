#!/usr/bin/env python3
"""Main test runner for SymbolicAD validation suite."""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

TESTS = [
    ("test_length_regression.py", "Length-Only Regression Test"),
    ("test_grounding_3tier.py", "3-Tier Grounding Evaluation"),
    ("test_spro_evaluation.py", "SPRO Evaluation Framework"),
]


def run_test(test_file: str, name: str) -> Tuple[bool, str]:
    """Run a single test and return (success, output)."""
    test_path = Path(__file__).parent / test_file
    if not test_path.exists():
        return False, f"Test file not found: {test_path}"
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(__file__).parent.parent,
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Test timed out (300s)"
    except Exception as e:
        return False, f"Error running test: {e}"


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("SymbolicAD Validation Suite")
    print("=" * 60)
    
    results = []
    for test_file, name in TESTS:
        print(f"\n--- Running {name} ---")
        success, output = run_test(test_file, name)
        results.append((name, success, output))
        
        if success:
            print(f"  ✓ {name} PASSED")
        else:
            print(f"  ✗ {name} FAILED")
            # Print first 500 chars of error
            print(f"  Error: {output[:500]}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success, _ in results:
        status = "PASS" if success else "FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(s for _, s, _ in results)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())