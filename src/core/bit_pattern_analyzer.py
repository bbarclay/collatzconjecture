"""
Bit pattern analysis tools for the Collatz conjecture proof.
This module implements analysis of dual-track bit length patterns.
"""

import math
from typing import List, Tuple, Dict, Optional
from .collatz_verifier import CollatzVerifier


class BitPatternAnalyzer:
    """Analyzes dual-track bit length patterns in Collatz steps"""

    def __init__(self):
        self.verifier = CollatzVerifier()

    def analyze_bit_patterns(
        self, n_range=range(1, 1000, 2)
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Analyze bit patterns for odd numbers"""
        upper_track = []  # Points on perfect prediction line
        lower_track = []  # Points below prediction line

        for n in n_range:
            initial_bits = len(format(n, "b"))
            next_n = self.verifier.collatz_step(n)
            actual_bits = len(format(next_n, "b"))
            predicted_bits = initial_bits + math.floor(math.log2(3))

            # Classify into tracks
            if actual_bits == predicted_bits:
                upper_track.append((initial_bits, actual_bits))
            else:
                lower_track.append((initial_bits, actual_bits))

            # Analyze spacing between tracks
            if len(upper_track) > 0 and len(lower_track) > 0:
                track_spacing = self.compute_track_spacing(
                    upper_track[-1], lower_track[-1]
                )

            # Check for 3n+1 relationship
            if n % 3 == 2:  # Special residue class
                self.analyze_residue_pattern(n, actual_bits, predicted_bits)

        return upper_track, lower_track

    def compute_track_spacing(
        self, upper_point: Tuple[int, int], lower_point: Tuple[int, int]
    ) -> Optional[int]:
        """Analyze vertical spacing between tracks"""
        if upper_point[0] == lower_point[0]:  # Same x-coordinate
            return upper_point[1] - lower_point[1]
        return None

    def analyze_residue_pattern(self, n: int, actual: int, predicted: int) -> Dict:
        """Analyze patterns for numbers mod 3"""
        tau = self.verifier.find_tau(n)
        compression = predicted - actual
        return {"n": n, "tau": tau, "compression": compression, "residue": n % 3}

    def find_special_sequences(self) -> List[int]:
        """Find sequences that maximize compression"""
        special_cases = []
        for n in range(1, 1000, 2):
            if self.is_optimal_compression(n):
                special_cases.append(n)
        return special_cases

    def is_optimal_compression(self, n: int) -> bool:
        """Check if n achieves maximum possible compression"""
        next_n = self.verifier.collatz_step(n)
        actual_bits = len(format(next_n, "b"))
        predicted_bits = len(format(n, "b")) + math.floor(math.log2(3))
        return (predicted_bits - actual_bits) > 1
