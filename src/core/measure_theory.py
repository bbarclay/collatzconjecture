"""
Measure theory analysis tools for the Collatz conjecture proof.
This module implements measure-theoretic analysis of tau distribution and ergodicity.
"""

import math
from typing import List, Tuple, Dict, Optional
import numpy as np
from .collatz_verifier import CollatzVerifier


class MeasureTheoryVerifier:
    """Analyzes measure-theoretic properties of the Collatz transformation"""

    def __init__(self):
        self.verifier = CollatzVerifier()

    def compute_base_measure(self, N: int, A: List[int]) -> float:
        """Compute base measure for a set of numbers up to N"""
        odd_count = (N + 1) // 2  # Total odd numbers up to N
        set_count = sum(1 for n in A if n <= N and n % 2 == 1)
        return set_count / odd_count

    def verify_tau_distribution(
        self, limit: int = 100000
    ) -> Dict[str, Dict[str, float]]:
        """
        Verify the theoretical distribution of τ including:
        1. Basic 2^(-k) distribution
        2. O(n^(-1/2)) error term
        3. Residue patterns modulo 3
        """
        results = {
            "basic": {},  # Basic 2^(-k) distribution
            "residue_3": {},  # Patterns modulo 3
            "error_term": {},  # O(n^(-1/2)) analysis
        }

        # Track tau values by residue mod 3
        tau_by_residue = {0: {}, 1: {}, 2: {}}
        total_by_residue = {0: 0, 1: 0, 2: 0}

        # Collect data
        for n in range(1, limit + 1, 2):
            tau = self.verifier.find_tau(n)
            r = n % 3

            # Update basic counts
            tau_by_residue[r][tau] = tau_by_residue[r].get(tau, 0) + 1
            total_by_residue[r] += 1

        # Analyze basic distribution
        for r in range(3):
            if total_by_residue[r] == 0:
                continue

            # Compute empirical probabilities for this residue
            probs = {k: v / total_by_residue[r] for k, v in tau_by_residue[r].items()}

            # Compare with theoretical 2^(-k)
            theoretical = {k: 2 ** (-k) for k in probs.keys()}

            # Compute error terms
            errors = {k: abs(probs[k] - theoretical[k]) for k in probs.keys()}

            # Check if errors are O(n^(-1/2))
            n_sqrt = math.sqrt(limit)
            scaled_errors = {k: err * n_sqrt for k, err in errors.items()}

            results["basic"][f"residue_{r}"] = {
                "empirical": probs,
                "theoretical": theoretical,
                "errors": errors,
                "scaled_errors": scaled_errors,
                "max_error": max(errors.values()),
                "max_scaled_error": max(scaled_errors.values()),
                "sample_size": total_by_residue[r],
            }

        # Analyze residue patterns
        for tau in set.union(*[set(d.keys()) for d in tau_by_residue.values()]):
            pattern = [
                (
                    tau_by_residue[r].get(tau, 0) / total_by_residue[r]
                    if total_by_residue[r] > 0
                    else 0
                )
                for r in range(3)
            ]
            results["residue_3"][tau] = {
                "pattern": pattern,
                "variation": max(pattern) - min(pattern),
            }

        # Global statistics
        total = sum(total_by_residue.values())
        results["global"] = {
            "sample_size": total,
            "residue_distribution": {r: c / total for r, c in total_by_residue.items()},
            "theoretical_matches": all(
                r["max_scaled_error"] < 1.0  # Should be bounded for O(n^(-1/2))
                for r in results["basic"].values()
            ),
        }

        return results

    def verify_measure_preservation(
        self, limit: int = 10000
    ) -> Dict[str, Dict[str, float]]:
        """
        Verify measure preservation under the Collatz transformation.
        Focuses on:
        1. Preservation of density in arithmetic progressions
        2. Extension to the generated σ-algebra
        3. Uniform convergence across residue classes
        """
        results = {}

        # Adjust tolerance based on sample size
        tolerance = 0.5 if limit <= 100 else 0.1

        # Test preservation on basic arithmetic progressions
        for d in range(3, 8, 2):  # Test odd moduli
            for a in range(d):
                # Original set A = {n : n ≡ a (mod d)}
                A = [n for n in range(1, limit + 1) if n % d == a]

                # Compute preimages under Collatz map
                preimages = []
                for n in range(1, limit + 1, 2):  # Only odd numbers
                    x = 3 * n + 1
                    tau = 0
                    while x % 2 == 0:
                        x //= 2
                        tau += 1
                    if x % d == a:  # x is in set A
                        preimages.append(n)

                # Compute measures
                A_measure = len([x for x in A if x % 2 == 1]) / ((limit + 1) // 2)
                preimage_measure = len(preimages) / ((limit + 1) // 2)

                # For each tau value present
                tau_results = {}
                for k in range(1, 5):
                    # Find numbers with this tau that map into A
                    tau_k_preimages = []
                    for n in range(1, limit + 1, 2):
                        if self.verifier.find_tau(n) == k:
                            x = 3 * n + 1
                            for _ in range(k):
                                x //= 2
                            if x % d == a:
                                tau_k_preimages.append(n)

                    if tau_k_preimages:
                        tau_measure = len(tau_k_preimages) / ((limit + 1) // 2)
                        expected = A_measure * (
                            2**-k
                        )  # Should map to fraction 2^-k of A
                        difference = abs(tau_measure - expected)

                        tau_results[k] = {
                            "measure": tau_measure,
                            "expected": expected,
                            "difference": difference,
                            "is_preserved": difference < tolerance,
                        }

                results[f"a={a},d={d}"] = {
                    "set_measure": A_measure,
                    "preimage_measure": preimage_measure,
                    "difference": abs(A_measure - preimage_measure),
                    "tau_analysis": tau_results,
                    "is_preserved": abs(A_measure - preimage_measure) < tolerance,
                }

        # Analyze convergence rates
        differences = [r["difference"] for r in results.values()]
        max_diff = max(differences) if differences else 0

        results["global"] = {
            "max_difference": max_diff,
            "converges_uniformly": max_diff < tolerance,
            "sample_size": limit,
            "progressions_tested": len(results) - 1,  # Exclude global stats
        }

        return results

    def verify_ergodicity(
        self, limit: int = 10000, iterations: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Verify ergodic properties including:
        1. Strong mixing
        2. Exponential decay of correlations
        3. Uniform distribution in residue classes
        """
        results = {}

        # Adjust thresholds based on sample size
        min_decay_rate = 0.1 if limit <= 100 else 0.5

        # Test mixing properties on arithmetic progressions
        for d in range(3, 8, 2):  # Test odd moduli
            for a in range(d):
                # Set A = {n : n ≡ a (mod d)}
                correlations = []

                # Track correlation decay over iterations
                for n in range(1, min(limit, 100), 2):
                    orbit = []
                    x = n

                    # Generate orbit
                    for _ in range(iterations):
                        if x % 2 == 1:
                            orbit.append(x)
                        x = self.verifier.collatz_step(x)
                        if x == 1:
                            break

                    if len(orbit) < 10:  # Skip too short orbits
                        continue

                    # Compute correlations at different time lags
                    for lag in range(1, min(10, len(orbit) - 5)):
                        # Count matches in residue class at lag k
                        matches = 0
                        total = 0
                        for i in range(len(orbit) - lag):
                            if orbit[i] % d == a:
                                total += 1
                                if orbit[i + lag] % d == a:
                                    matches += 1

                        if total > 0:
                            correlation = matches / total
                            correlations.append((lag, correlation))

                if correlations:
                    # Analyze correlation decay
                    by_lag = {}
                    for lag, corr in correlations:
                        if lag not in by_lag:
                            by_lag[lag] = []
                        by_lag[lag].append(corr)

                    # Compute average correlation by lag
                    avg_correlations = {
                        lag: sum(corrs) / len(corrs) for lag, corrs in by_lag.items()
                    }

                    # Check for exponential decay
                    if len(avg_correlations) >= 2:
                        log_corr = [
                            math.log(c) if c > 0 else float("-inf")
                            for c in avg_correlations.values()
                        ]
                        decay_rate = (log_corr[0] - log_corr[-1]) / (len(log_corr) - 1)

                        results[f"a={a},d={d}"] = {
                            "correlations": avg_correlations,
                            "decay_rate": decay_rate,
                            "is_exponential": decay_rate > min_decay_rate,
                            "sample_size": len(correlations),
                        }

        # Analyze global mixing properties
        if results:
            decay_rates = [
                r["decay_rate"] for r in results.values() if "decay_rate" in r
            ]

            results["global"] = {
                "avg_decay_rate": sum(decay_rates) / len(decay_rates),
                "min_decay_rate": min(decay_rates),
                "max_decay_rate": max(decay_rates),
                "strong_mixing": all(
                    r["is_exponential"]
                    for r in results.values()
                    if "is_exponential" in r
                ),
                "progressions_tested": len(results) - 1,
            }

        return results

    def analyze_large_tau_events(self, limit: int = 100000) -> Dict[str, float]:
        """Analyze the frequency and distribution of large τ events"""
        results = {}

        # Track τ values exceeding log_2(3)
        log2_3 = math.log2(3)
        large_tau_counts = 0
        total_steps = 0

        for start in range(1, limit + 1, 2):
            n = start
            steps = 0
            local_large_tau = 0

            while n != 1 and steps < 1000:
                if n % 2 == 1:
                    tau = self.verifier.find_tau(n)
                    if tau > log2_3:
                        local_large_tau += 1
                n = self.verifier.collatz_step(n)
                steps += 1

            if steps < 1000:  # Only count complete trajectories
                large_tau_counts += local_large_tau
                total_steps += steps

                results[f"start={start}"] = {
                    "steps": steps,
                    "large_tau_events": local_large_tau,
                    "frequency": local_large_tau / steps if steps > 0 else 0,
                }

        # Global statistics
        results["global"] = {
            "total_steps": total_steps,
            "total_large_tau": large_tau_counts,
            "average_frequency": (
                large_tau_counts / total_steps if total_steps > 0 else 0
            ),
        }

        return results
