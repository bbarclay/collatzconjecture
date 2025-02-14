"""
Information theory analysis tools for the Collatz conjecture proof.
This module implements entropy and information loss analysis.
"""

import math
from typing import List, Tuple, Dict, Optional
from .collatz_verifier import CollatzVerifier


class InformationTheoryVerifier:
    """Analyzes entropy and information loss in Collatz transformations"""

    def __init__(self):
        self.verifier = CollatzVerifier()

    def entropy_change(self, n: int) -> Dict[str, float]:
        """Calculate detailed entropy change for one step"""
        if n % 2 == 0:
            raise ValueError("n must be odd")

        tau = self.verifier.find_tau(n)
        next_n = self.verifier.collatz_step(n)

        # Actual entropy change
        h1 = self.binary_entropy(n)
        h2 = self.binary_entropy(next_n)
        delta_h = h2 - h1

        # Theoretical prediction
        theoretical = math.log2(3) - tau
        error = delta_h - theoretical

        # Bit analysis
        actual_bits = len(format(next_n, "b"))
        predicted_bits = len(format(n, "b")) + math.floor(math.log2(3)) - tau

        return {
            "delta_h": delta_h,
            "theoretical": theoretical,
            "error": error,
            "tau": tau,
            "actual_bits": actual_bits,
            "predicted_bits": predicted_bits,
        }

    def binary_entropy(self, x: int) -> float:
        """Calculate binary entropy of a number"""
        return math.log2(x)

    def analyze_information_loss(self, n: int) -> Dict[str, float]:
        """Analyze information loss in one step"""
        if n % 2 == 0:
            raise ValueError("n must be odd")

        tau = self.verifier.find_tau(n)
        next_n = self.verifier.collatz_step(n)

        # Information added by multiplication by 3
        info_added = math.log2(3)

        # Information lost by division by 2^tau
        info_lost = tau

        # Net information change
        net_change = info_added - info_lost

        # Error term
        error_term = math.log2(1 + 1 / (3 * n))

        return {
            "info_added": info_added,
            "info_lost": info_lost,
            "net_change": net_change,
            "error_term": error_term,
            "tau": tau,
        }

    def verify_entropy_bounds(self, n: int) -> Dict[str, float]:
        """Verify theoretical entropy change bounds"""
        if n % 2 == 0:
            raise ValueError("n must be odd")

        changes = self.entropy_change(n)

        # Theoretical bounds
        upper_bound = math.log2(3) - changes["tau"]
        lower_bound = upper_bound - 1 / (3 * n * math.log(2))

        # Check if actual change is within bounds
        actual = changes["delta_h"]
        within_bounds = lower_bound <= actual <= upper_bound

        return {
            "actual": actual,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "within_bounds": within_bounds,
            "margin": min(upper_bound - actual, actual - lower_bound),
        }

    def analyze_trajectory_entropy(
        self, start: int, max_steps: int = 100
    ) -> List[Dict[str, float]]:
        """Analyze entropy changes along a trajectory"""
        path, _ = self.verifier.trajectory(start)
        changes = []

        for i in range(len(path) - 1):
            if path[i] % 2 == 1:  # Only analyze odd steps
                change = self.entropy_change(path[i])
                bounds = self.verify_entropy_bounds(path[i])
                info_loss = self.analyze_information_loss(path[i])

                changes.append(
                    {
                        "step": i,
                        "n": path[i],
                        "next_n": path[i + 1],
                        "entropy_change": change,
                        "bounds": bounds,
                        "information_loss": info_loss,
                    }
                )

        return changes
