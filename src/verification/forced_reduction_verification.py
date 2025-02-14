#!/usr/bin/env python3

"""
Forced Reduction Verification
============================

This module provides comprehensive verification of the forced reduction properties
of the Collatz function, supporting the theoretical claims in the paper.
"""

import math
import statistics
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np


class ForcedReductionAnalyzer:
    """Analyzes and verifies forced reduction properties"""

    def __init__(self, max_steps: int = 1000000):
        self.max_steps = max_steps
        self.cache: Dict[int, Any] = {}

    def analyze_trajectory(self, n: int) -> Dict[str, Any]:
        """Full trajectory analysis with tau events"""
        trajectory: List[int] = []
        tau_events: List[Tuple[int, int, int]] = []
        current = n
        steps = 0
        max_value = n

        while current != 1 and steps < self.max_steps:
            trajectory.append(current)
            if current % 2 == 1:
                tau = self.find_tau(current)
                tau_events.append((steps, current, tau))

            current = self.collatz_step(current)
            max_value = max(max_value, current)
            steps += 1

        return {
            "steps": steps,
            "max_value": max_value,
            "trajectory": trajectory,
            "tau_events": tau_events,
            "converged": current == 1,
            "entropy_changes": self.calculate_entropy_changes(trajectory),
        }

    def find_tau(self, n: int) -> int:
        """Compute tau(n) for odd n"""
        if n % 2 == 0:
            raise ValueError("n must be odd")
        x = 3 * n + 1
        tau = 0
        while x % 2 == 0:
            tau += 1
            x //= 2
        return tau

    def collatz_step(self, n: int) -> int:
        """Single Collatz step"""
        return n // 2 if n % 2 == 0 else 3 * n + 1

    def calculate_entropy_changes(self, trajectory: List[int]) -> List[float]:
        """Calculate entropy changes between consecutive steps"""
        changes = []
        for i in range(len(trajectory) - 1):
            n1, n2 = trajectory[i], trajectory[i + 1]
            h1 = math.log2(n1) if n1 > 0 else 0
            h2 = math.log2(n2) if n2 > 0 else 0
            changes.append(h2 - h1)
        return changes

    def analyze_bit_patterns(self, n: int) -> List[Dict[str, Any]]:
        """Analyze bit pattern evolution"""
        patterns = []
        current = n
        while current != 1 and len(patterns) < 100:
            if current % 2 == 1:
                binary = format(current, "b")
                next_n = self.collatz_step(current)
                next_binary = format(next_n, "b")
                hamming_dist = sum(
                    b1 != b2
                    for b1, b2 in zip(binary.zfill(len(next_binary)), next_binary)
                )
                patterns.append(
                    {
                        "value": current,
                        "binary": binary,
                        "next_value": next_n,
                        "next_binary": next_binary,
                        "tau": self.find_tau(current),
                        "hamming_distance": hamming_dist,
                        "bit_length_change": len(next_binary) - len(binary),
                    }
                )
            current = self.collatz_step(current)
        return patterns

    def verify_residue_classes(self, limit: int = 1000) -> Dict[int, Dict[str, float]]:
        """Verify tau distribution in residue classes mod 3"""
        residue_stats = {0: [], 1: [], 2: []}
        for n in range(1, limit + 1, 2):
            residue = n % 3
            tau = self.find_tau(n)
            residue_stats[residue].append(tau)

        return {
            r: {
                "mean": statistics.mean(taus),
                "median": statistics.median(taus),
                "stdev": statistics.stdev(taus) if len(taus) > 1 else 0,
                "max": max(taus),
                "min": min(taus),
            }
            for r, taus in residue_stats.items()
        }

    def analyze_mersenne(self, k_values=range(3, 20)) -> List[Dict[str, Any]]:
        """Analyze behavior of Mersenne numbers"""
        results = []
        for k in k_values:
            n = 2**k - 1
            analysis = self.analyze_trajectory(n)
            results.append(
                {
                    "k": k,
                    "n": n,
                    "steps": analysis["steps"],
                    "max_value": analysis["max_value"],
                    "tau_events": analysis["tau_events"],
                    "entropy_changes": analysis["entropy_changes"],
                }
            )
        return results

    def verify_alternating_patterns(self, length: int = 10) -> List[Dict[str, Any]]:
        """Verify behavior of alternating bit patterns"""
        pattern = int("10" * length, 2)
        return self.analyze_bit_patterns(pattern)

    def plot_trajectory_stats(self, n: int, save_path: str = None):
        """Plot trajectory statistics"""
        analysis = self.analyze_trajectory(n)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Plot trajectory values
        steps = range(len(analysis["trajectory"]))
        ax1.plot(steps, analysis["trajectory"])
        ax1.set_yscale("log")
        ax1.set_title(f"Trajectory for n={n}")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Value")

        # Plot tau events
        if analysis["tau_events"]:
            steps, values, taus = zip(*analysis["tau_events"])
            ax2.scatter(steps, taus)
            ax2.set_title("Tau Events")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Tau Value")

        # Plot entropy changes
        if analysis["entropy_changes"]:
            ax3.plot(
                range(len(analysis["entropy_changes"])), analysis["entropy_changes"]
            )
            ax3.set_title("Entropy Changes")
            ax3.set_xlabel("Step")
            ax3.set_ylabel("Entropy Change")
            ax3.axhline(y=0, color="r", linestyle="--")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()


def verify_forced_reduction_claims():
    """Verify all major claims about forced reduction"""
    analyzer = ForcedReductionAnalyzer()

    # 1. Verify tau distribution
    print("=== Tau Distribution Analysis ===")
    residue_stats = analyzer.verify_residue_classes(10000)
    for r, stats in residue_stats.items():
        print(f"\nn ≡ {r} (mod 3):")
        for stat, value in stats.items():
            print(f"  {stat}: {value:.2f}")

    # 2. Analyze Mersenne numbers
    print("\n=== Mersenne Number Analysis ===")
    mersenne_results = analyzer.analyze_mersenne()
    for result in mersenne_results:
        tau_values = [tau for _, _, tau in result["tau_events"]]
        print(f"\nM_{result['k']} = {result['n']}:")
        print(f"  Steps: {result['steps']}")
        print(f"  Max value: {result['max_value']}")
        if tau_values:
            print(f"  Average tau: {sum(tau_values)/len(tau_values):.2f}")
            print(f"  Max tau: {max(tau_values)}")

    # 3. Test alternating bit patterns
    print("\n=== Alternating Bit Pattern Analysis ===")
    alt_patterns = analyzer.verify_alternating_patterns()
    for i, step in enumerate(alt_patterns[:5]):
        print(f"\nStep {i+1}:")
        print(f"  Value: {step['value']}")
        print(f"  Binary: {step['binary']}")
        print(f"  Tau: {step['tau']}")
        print(f"  Hamming distance: {step['hamming_distance']}")
        print(f"  Bit length change: {step['bit_length_change']}")

    # 4. Large number trajectory analysis
    print("\n=== Large Number Analysis ===")
    large_numbers = [2**k - 1 for k in [50, 75, 100]]
    for n in large_numbers:
        print(f"\nAnalyzing n = 2^{int(math.log2(n+1))} - 1:")
        analysis = analyzer.analyze_trajectory(n)
        tau_stats = [tau for _, _, tau in analysis["tau_events"]]
        print(f"  Steps: {analysis['steps']}")
        print(f"  Max value: {analysis['max_value']}")
        if tau_stats:
            print(f"  Average tau: {sum(tau_stats)/len(tau_stats):.2f}")
            print(f"  Max tau: {max(tau_stats)}")
            print(
                f"  Entropy changes (avg): {sum(analysis['entropy_changes'])/len(analysis['entropy_changes']):.2f}"
            )

        # Generate visualization
        analyzer.plot_trajectory_stats(n, f"trajectory_{int(math.log2(n+1))}.png")


if __name__ == "__main__":
    verify_forced_reduction_claims()
