"""
Visualization tools for the Collatz conjecture proof.
This module implements plotting and visualization functionality.
"""

import math
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .collatz_verifier import CollatzVerifier
from .information_theory import InformationTheoryVerifier
from .measure_theory import MeasureTheoryVerifier


class CollatzVisualizer:
    """Generates visualizations for Collatz conjecture analysis"""

    def __init__(self):
        self.verifier = CollatzVerifier()
        self.info_verifier = InformationTheoryVerifier()
        self.measure_verifier = MeasureTheoryVerifier()

        # Set style defaults
        plt.style.use("seaborn")
        sns.set_palette("husl")

    def plot_trajectory(self, start: int, save_path: Optional[str] = None) -> None:
        """Plot Collatz trajectory with entropy changes"""
        path, _ = self.verifier.trajectory(start)
        steps = range(len(path))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot trajectory
        ax1.plot(steps, path, "b-", label="Value")
        ax1.set_yscale("log")
        ax1.set_title(f"Collatz Trajectory for n={start}")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Value")
        ax1.grid(True)
        ax1.legend()

        # Plot entropy changes
        changes = self.info_verifier.analyze_trajectory_entropy(start)
        if changes:
            steps = [c["step"] for c in changes]
            deltas = [c["entropy_change"]["delta_h"] for c in changes]
            ax2.plot(steps, deltas, "r-", label="Entropy Change")
            ax2.axhline(y=0, color="k", linestyle="--")
            ax2.set_title("Entropy Changes")
            ax2.set_xlabel("Steps")
            ax2.set_ylabel("ΔH")
            ax2.grid(True)
            ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_tau_distribution(
        self, limit: int = 100000, save_path: Optional[str] = None
    ) -> None:
        """Plot empirical vs theoretical τ distribution"""
        results = self.measure_verifier.verify_tau_distribution(limit)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Distribution comparison
        k_values = sorted(results["empirical"].keys())
        empirical = [results["empirical"][k] for k in k_values]
        theoretical = [results["theoretical"][k] for k in k_values]

        ax1.bar(k_values, empirical, alpha=0.5, label="Empirical")
        ax1.plot(k_values, theoretical, "r-", label="Theoretical (2^-k)")
        ax1.set_title("τ Distribution")
        ax1.set_xlabel("τ Value")
        ax1.set_ylabel("Probability")
        ax1.legend()
        ax1.grid(True)

        # Error analysis
        errors = [results["errors"][k] for k in k_values]
        ax2.bar(k_values, errors)
        ax2.set_title("Distribution Error")
        ax2.set_xlabel("τ Value")
        ax2.set_ylabel("|Empirical - Theoretical|")
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_bit_patterns(
        self, limit: int = 1000, save_path: Optional[str] = None
    ) -> None:
        """Plot dual-track bit length patterns"""
        from .bit_pattern_analyzer import BitPatternAnalyzer

        analyzer = BitPatternAnalyzer()
        upper_track, lower_track = analyzer.analyze_bit_patterns(range(1, limit, 2))

        plt.figure(figsize=(12, 8))

        # Plot upper track points
        if upper_track:
            x_upper, y_upper = zip(*upper_track)
            plt.scatter(x_upper, y_upper, c="blue", label="Upper Track", alpha=0.5)

        # Plot lower track points
        if lower_track:
            x_lower, y_lower = zip(*lower_track)
            plt.scatter(x_lower, y_lower, c="red", label="Lower Track", alpha=0.5)

        # Perfect prediction line
        x_range = range(
            min(min(x_upper), min(x_lower)), max(max(x_upper), max(x_lower)) + 1
        )
        y_pred = [x + math.floor(math.log2(3)) for x in x_range]
        plt.plot(x_range, y_pred, "k--", label="Perfect Prediction")

        plt.title("Dual-Track Bit Length Evolution")
        plt.xlabel("Input Bit Length")
        plt.ylabel("Output Bit Length")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_ergodicity_analysis(
        self, limit: int = 10000, save_path: Optional[str] = None
    ) -> None:
        """Plot ergodicity analysis results"""
        results = self.measure_verifier.verify_ergodicity(limit)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot orbit lengths
        starts = []
        lengths = []
        for key, value in results.items():
            if key.startswith("start="):
                start = int(key.split("=")[1])
                starts.append(start)
                lengths.append(value["orbit_length"])

        ax1.bar(starts, lengths)
        ax1.set_title("Orbit Lengths")
        ax1.set_xlabel("Starting Value")
        ax1.set_ylabel("Orbit Length")
        ax1.grid(True)

        # Plot measure convergence
        if results and "start=1" in results:
            measures = results["start=1"]["measures"]
            sets = list(measures.keys())
            values = list(measures.values())

            ax2.bar(range(len(sets)), values)
            ax2.set_xticks(range(len(sets)))
            ax2.set_xticklabels(sets, rotation=45)
            ax2.set_title("Measure Distribution (n=1)")
            ax2.set_ylabel("Empirical Measure")
            ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_large_tau_analysis(
        self, limit: int = 100000, save_path: Optional[str] = None
    ) -> None:
        """Plot analysis of large τ events"""
        results = self.measure_verifier.analyze_large_tau_events(limit)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot frequency distribution
        frequencies = []
        for key, value in results.items():
            if key != "global" and value["steps"] > 0:
                frequencies.append(value["frequency"])

        ax1.hist(frequencies, bins=50, density=True)
        ax1.axvline(
            results["global"]["average_frequency"],
            color="r",
            linestyle="--",
            label="Global Average",
        )
        ax1.set_title("Distribution of Large τ Frequencies")
        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(True)

        # Plot cumulative statistics
        steps = []
        events = []
        for key, value in results.items():
            if key != "global":
                steps.append(value["steps"])
                events.append(value["large_tau_events"])

        ax2.scatter(steps, events, alpha=0.5)
        ax2.set_title("Large τ Events vs Trajectory Length")
        ax2.set_xlabel("Steps to 1")
        ax2.set_ylabel("Number of Large τ Events")
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
