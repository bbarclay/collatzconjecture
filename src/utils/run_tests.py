#!/usr/bin/env python3
"""
Test runner for Collatz conjecture verification code.
Executes all tests and generates a comprehensive report.
"""

import unittest
import sys
import time
from datetime import datetime
from pathlib import Path
import json


def run_tests():
    """Run all tests and return results"""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = "tests"
    suite = loader.discover(start_dir, pattern="test_*.py")

    # Prepare results collection
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0},
        "details": [],
    }

    # Run tests with result collection
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    test_results = runner.run(suite)
    end_time = time.time()

    # Collect results
    results["tests"]["total"] = test_results.testsRun
    results["tests"]["passed"] = (
        test_results.testsRun - len(test_results.failures) - len(test_results.errors)
    )
    results["tests"]["failed"] = len(test_results.failures)
    results["tests"]["errors"] = len(test_results.errors)
    results["tests"]["skipped"] = len(test_results.skipped)
    results["execution_time"] = end_time - start_time

    # Collect failure details
    for failure in test_results.failures:
        results["details"].append(
            {"test": str(failure[0]), "type": "failure", "message": failure[1]}
        )

    for error in test_results.errors:
        results["details"].append(
            {"test": str(error[0]), "type": "error", "message": error[1]}
        )

    return results


def save_results(results):
    """Save test results to a file"""
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"test_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return output_file


def print_summary(results):
    """Print a summary of test results"""
    print("\n=== Test Execution Summary ===")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total tests: {results['tests']['total']}")
    print(f"Passed: {results['tests']['passed']}")
    print(f"Failed: {results['tests']['failed']}")
    print(f"Errors: {results['tests']['errors']}")
    print(f"Skipped: {results['tests']['skipped']}")
    print(f"Execution time: {results['execution_time']:.2f} seconds")

    if results["details"]:
        print("\n=== Failed Tests ===")
        for detail in results["details"]:
            print(f"\nTest: {detail['test']}")
            print(f"Type: {detail['type']}")
            print("Message:")
            print(detail["message"])


if __name__ == "__main__":
    # Run tests and collect results
    results = run_tests()

    # Save results to file
    output_file = save_results(results)

    # Print summary
    print_summary(results)
    print(f"\nDetailed results saved to: {output_file}")

    # Exit with appropriate code
    sys.exit(len(results["details"]) > 0)
