#!/usr/bin/env python3
"""
Main verification script for the Collatz conjecture proof.
This script orchestrates the verification of all theorems and properties.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from src.collatz_verifier import CollatzVerifier
from src.bit_pattern_analyzer import BitPatternAnalyzer
from src.information_theory import InformationTheoryVerifier
from src.measure_theory import MeasureTheoryVerifier
from src.theorem_mapper import TheoremMapper
from tests.config import TEST_PARAMS, TEST_DATA, LOG_CONFIG


class ProofVerifier:
    """Orchestrates verification of the entire Collatz proof"""

    def __init__(self):
        # Initialize verifiers
        self.collatz = CollatzVerifier()
        self.bit_analyzer = BitPatternAnalyzer()
        self.info_verifier = InformationTheoryVerifier()
        self.measure_verifier = MeasureTheoryVerifier()
        self.theorem_mapper = TheoremMapper()

        # Set up logging
        self._setup_logging()

        # Initialize results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "verification_results": {},
            "statistics": {},
            "errors": [],
        }

    def _setup_logging(self):
        """Configure logging"""
        log_dir = Path(TEST_DATA["log_dir"])
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"verification_{timestamp}.log"

        logging.basicConfig(
            level=LOG_CONFIG["level"],
            format=LOG_CONFIG["format"],
            datefmt=LOG_CONFIG["date_format"],
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger("ProofVerifier")

    def verify_cryptographic_framework(self) -> Dict[str, Any]:
        """Verify cryptographic framework properties"""
        self.logger.info("Verifying cryptographic framework...")
        results = {}

        # Verify one-way property
        results["one_way"] = self.theorem_mapper.verify_theorem("thm:one_way")

        # Verify avalanche effect
        results["avalanche"] = self.theorem_mapper.verify_theorem("thm:avalanche")

        # Verify entropy reduction
        results["entropy"] = self.theorem_mapper.verify_theorem("thm:entropy")

        return results

    def verify_measure_theory(self) -> Dict[str, Any]:
        """Verify measure-theoretic properties"""
        self.logger.info("Verifying measure-theoretic properties...")
        results = {}

        # Verify τ distribution
        results["tau_distribution"] = self.theorem_mapper.verify_theorem("thm:tau_dist")

        # Verify measure preservation
        results["measure_preservation"] = self.theorem_mapper.verify_theorem(
            "thm:measure_preserve"
        )

        # Verify ergodicity
        results["ergodicity"] = self.theorem_mapper.verify_theorem("thm:ergodic")

        return results

    def verify_information_theory(self) -> Dict[str, Any]:
        """Verify information-theoretic properties"""
        self.logger.info("Verifying information-theoretic properties...")
        results = {}

        # Verify entropy bounds
        limit = TEST_PARAMS["medium_sample"]
        for n in range(1, limit, 2):
            bounds = self.info_verifier.verify_entropy_bounds(n)
            if not bounds["within_bounds"]:
                self.logger.warning(f"Entropy bounds violated for n={n}")
                results.setdefault("violations", []).append({"n": n, "bounds": bounds})

        # Analyze information loss
        results["info_loss"] = {
            "statistics": {},
            "large_tau_events": self.measure_verifier.analyze_large_tau_events(limit),
        }

        return results

    def verify_global_behavior(self) -> Dict[str, Any]:
        """Verify global behavior properties"""
        self.logger.info("Verifying global behavior...")
        results = {}

        # Verify cycle prevention
        results["cycle_prevention"] = self.theorem_mapper.verify_theorem(
            "thm:cycle_prevent"
        )

        # Verify global descent
        results["global_descent"] = self.theorem_mapper.verify_theorem(
            "thm:global_descent"
        )

        return results

    def run_verification(self) -> Dict[str, Any]:
        """Run complete verification of the proof"""
        self.logger.info("Starting complete proof verification...")

        try:
            # Verify each component
            self.results["verification_results"][
                "cryptographic"
            ] = self.verify_cryptographic_framework()
            self.results["verification_results"][
                "measure_theory"
            ] = self.verify_measure_theory()
            self.results["verification_results"][
                "information_theory"
            ] = self.verify_information_theory()
            self.results["verification_results"][
                "global_behavior"
            ] = self.verify_global_behavior()

            # Compute statistics
            self.results["statistics"] = self._compute_verification_statistics()

            self.logger.info("Verification completed successfully")

        except Exception as e:
            self.logger.error(f"Verification failed: {str(e)}", exc_info=True)
            self.results["errors"].append(str(e))

        return self.results

    def _compute_verification_statistics(self) -> Dict[str, Any]:
        """Compute overall verification statistics"""
        stats = {
            "total_theorems": len(self.theorem_mapper.theorems),
            "verified_theorems": 0,
            "failed_theorems": 0,
            "verification_coverage": 0.0,
        }

        # Count verified theorems
        for section in self.results["verification_results"].values():
            for result in section.values():
                if isinstance(result, dict) and not result.get("errors"):
                    stats["verified_theorems"] += 1
                else:
                    stats["failed_theorems"] += 1

        # Compute coverage
        stats["verification_coverage"] = (
            stats["verified_theorems"] / stats["total_theorems"]
            if stats["total_theorems"] > 0
            else 0.0
        )

        return stats

    def save_results(self) -> Path:
        """Save verification results to file"""
        results_dir = Path(TEST_DATA["results_dir"])
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"verification_results_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        self.logger.info(f"Results saved to {output_file}")
        return output_file


def main():
    """Main entry point"""
    verifier = ProofVerifier()
    results = verifier.run_verification()
    output_file = verifier.save_results()

    # Print summary
    print("\n=== Verification Summary ===")
    print(f"Total theorems: {results['statistics']['total_theorems']}")
    print(f"Verified theorems: {results['statistics']['verified_theorems']}")
    print(f"Failed theorems: {results['statistics']['failed_theorems']}")
    print(
        f"Verification coverage: {results['statistics']['verification_coverage']*100:.1f}%"
    )

    if results["errors"]:
        print("\nErrors encountered:")
        for error in results["errors"]:
            print(f"- {error}")

    print(f"\nDetailed results saved to: {output_file}")

    # Exit with appropriate code
    sys.exit(len(results["errors"]) > 0)


if __name__ == "__main__":
    main()
