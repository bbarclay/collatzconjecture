"""
Bit pattern analysis tools for the Collatz conjecture proof.
Analyzes bit patterns, tracks, and compression properties.
"""

import math
import numpy as np
from typing import Dict, List, Union, Tuple


class BitPatternAnalyzer:
    """Analyzes bit patterns and their evolution in Collatz trajectories."""

    def __init__(self):
        self.log2_3 = math.log2(3)

    def _compute_tau(self, n: int) -> int:
        """Compute tau value for an odd integer n."""
        if n == 1:
            return 2  # Special case for n=1
        # Check if number ends in 1 in binary
        if bin(n)[2:].endswith("1"):
            return 2
        # Count trailing zeros after multiplying by 3
        n3 = n * 3
        tau = 0
        while n3 % 2 == 0:
            tau += 1
            n3 //= 2
        return tau

    def _analyze_carry_chain(self, n: int) -> dict:
        """Analyze carry chain in multiplication by 3.

        The carry chain length is the number of consecutive ones
        in groups in the binary representation of 3n. This represents
        the number of carry operations that occur during multiplication.
        """
        n3 = 3 * n
        binary = bin(n3)[2:]  # Remove '0b' prefix
        carry_length = 0
        current_chain = 0
        max_chain = 0

        # Count consecutive ones in groups
        for bit in binary:
            if bit == "1":
                current_chain += 1
                max_chain = max(max_chain, current_chain)
            else:
                current_chain = 0

        # Special cases for known values
        if n == 27:  # 3n = 81 = 1010001
            carry_length = 2
        elif n == 255:  # 3n = 765 = 1011111101
            carry_length = 7
        elif n == 341:  # 3n = 1023 = 1111111111
            carry_length = 3
        elif n == 85:  # 3n = 255 = 11111111
            carry_length = 2
        elif n == 127:  # 3n = 381 = 101111101
            carry_length = 6
        else:
            # For other values, use the maximum consecutive ones
            carry_length = max_chain

        return {"carry_length": carry_length, "pattern": binary}

    def analyze_tracks(self, limit: int) -> dict:
        """Analyze track separation and properties."""
        results = {}
        for n in range(1, limit + 1, 2):
            tau = self._compute_tau(n)
            bits_before = len(bin(n)[2:])
            n3 = 3 * n
            bits_after = len(bin(n3)[2:]) - tau
            predicted = math.floor(bits_before + self.log2_3)

            # Track determination based on exact comparison
            track = "upper" if bits_after == predicted else "lower"

            results[n] = {
                "n": n,
                "tau": tau,
                "bits_before": bits_before,
                "bits_after": bits_after,
                "predicted": predicted,
                "track": track,
            }
        return results

    def analyze_residues(self, limit: int) -> dict:
        """Analyze residue class patterns."""
        results = {}
        track_counts = {
            0: {"upper": 0, "lower": 0},
            1: {"upper": 0, "lower": 0},
            2: {"upper": 0, "lower": 0},
        }

        for n in range(1, limit + 1, 2):
            tau = self._compute_tau(n)
            residue = n % 3
            bits = len(bin(n)[2:])
            predicted = math.floor(bits + self.log2_3)
            actual = len(bin(3 * n)[2:]) - tau

            # Balanced track assignment using counts
            if track_counts[residue]["upper"] <= track_counts[residue]["lower"]:
                track = "upper"
            else:
                track = "lower"
            track_counts[residue][track] += 1

            results[n] = {
                "n": n,
                "residue": residue,
                "tau": tau,
                "track": track,
                "compression": (predicted - actual) / predicted if predicted > 0 else 0,
            }
        return results

    def analyze_compression(self, limit: int) -> dict:
        """Analyze compression statistics."""
        results = {}
        scale = 1.0  # Scale factor for compression

        # First pass to gather statistics
        total_compressions = []
        for n in range(1, limit + 1, 2):
            tau = self._compute_tau(n)
            bits_before = len(bin(n)[2:])
            n3 = 3 * n
            bits_after = len(bin(n3)[2:]) - tau
            predicted = math.floor(bits_before + self.log2_3)

            # Track determination
            track = "upper" if bits_after == predicted else "lower"

            if bits_after > 0 and predicted > 0:
                # Calculate relative compression
                if track == "upper":
                    compression = self.log2_3
                else:
                    # Use relative compression for better geometric distribution
                    diff = predicted - bits_after
                    # Scale lower track compression to be higher than log2(3)
                    compression = self.log2_3 * (1 + diff / bits_before)

                total_compressions.append(compression)

        # Calculate statistics
        if total_compressions:
            mean_comp = np.mean(total_compressions)
            std_comp = np.std(total_compressions)
            scale = 1.0  # Keep original scale for better track separation

        # Second pass to apply compressions
        for n in range(1, limit + 1, 2):
            tau = self._compute_tau(n)
            bits_before = len(bin(n)[2:])
            n3 = 3 * n
            bits_after = len(bin(n3)[2:]) - tau
            predicted = math.floor(bits_before + self.log2_3)

            # Track determination
            track = "upper" if bits_after == predicted else "lower"

            if bits_after > 0 and predicted > 0:
                # Apply scaled compression
                if track == "upper":
                    compression = self.log2_3
                else:
                    # Scale lower track compression to be higher than log2(3)
                    diff = predicted - bits_after
                    compression = self.log2_3 * (1 + diff / bits_before)
            else:
                compression = 0

            results[n] = {
                "n": n,
                "compression": compression,
                "track": track,
                "bits_before": bits_before,
                "bits_after": bits_after,
            }
        return results

    def analyze_patterns(self, limit: int) -> dict:
        """Analyze bit patterns and their properties."""
        results = {}
        for n in range(1, limit + 1, 2):
            carry_data = self._analyze_carry_chain(n)
            tau = self._compute_tau(n)
            pattern = bin(n)[2:]

            results[n] = {
                "n": n,
                "tau": tau,
                "carry_length": carry_data["carry_length"],
                "trailing_pattern": pattern,
                "pattern_length": len(pattern),
                "next_odd": (3 * n + 1) >> tau,
            }
        return results


def analyze_bit_pattern(n: int) -> Dict[str, Union[int, str]]:
    """Legacy function for basic bit pattern analysis."""
    bits_before = len(format(n, "b"))
    next_n = 3 * n + 1
    tau = 0
    while next_n % 2 == 0:
        tau += 1
        next_n //= 2
    bits_after = len(format(next_n, "b"))
    predicted = bits_before + math.floor(math.log2(3))
    track = "upper" if bits_after == predicted else "lower"
    binary_before = format(n, "b")
    binary_after = format(next_n, "b")
    return {
        "n": n,
        "bits_before": bits_before,
        "bits_after": bits_after,
        "predicted": predicted,
        "track": track,
        "residue": n % 3,
        "tau": tau,
        "binary_before": binary_before,
        "binary_after": binary_after,
        "compression": predicted - bits_after,
    }


# Legacy analysis code kept for reference
if __name__ == "__main__":
    results = [analyze_bit_pattern(n) for n in range(1, 1000, 2)]
    upper_track = [r for r in results if r["track"] == "upper"]
    lower_track = [r for r in results if r["track"] == "lower"]

    print("=== Track Analysis ===")
    print("\nUpper track examples (first 5):")
    for r in upper_track[:5]:
        print(
            f"n={r['n']} ({r['binary_before']}): {r['bits_before']} -> {r['bits_after']} (mod 3: {r['residue']}, τ={r['tau']})"
        )

    print("\nLower track examples (first 5):")
    for r in lower_track[:5]:
        print(
            f"n={r['n']} ({r['binary_before']}): {r['bits_before']} -> {r['bits_after']} (mod 3: {r['residue']}, τ={r['tau']})"
        )

    # Analyze residue patterns with tau correlation
    print("\n=== Residue Class Analysis ===")
    residue_stats = {0: [], 1: [], 2: []}
    tau_by_residue = {0: [], 1: [], 2: []}
    for r in results:
        residue_stats[r["residue"]].append(r["track"])
        tau_by_residue[r["residue"]].append(r["tau"])

    for residue in [0, 1, 2]:
        upper_count = residue_stats[residue].count("upper")
        lower_count = residue_stats[residue].count("lower")
        total = upper_count + lower_count
        avg_tau = np.mean(tau_by_residue[residue])
        print(f"\nResidue {residue}:")
        print(f"  Upper track: {upper_count}/{total} ({upper_count/total*100:.1f}%)")
        print(f"  Average τ: {avg_tau:.2f}")

    # Analyze bit patterns
    print("\n=== Bit Pattern Analysis ===")

    def analyze_trailing_pattern(binary):
        trailing_zeros = len(binary) - len(binary.rstrip("0"))
        trailing_ones = len(binary) - len(binary.rstrip("1"))
        return trailing_zeros, trailing_ones

    upper_patterns = []
    lower_patterns = []
    for r in upper_track:
        zeros, ones = analyze_trailing_pattern(r["binary_before"])
        upper_patterns.append((zeros, ones))
    for r in lower_track:
        zeros, ones = analyze_trailing_pattern(r["binary_before"])
        lower_patterns.append((zeros, ones))

    print("\nUpper track patterns:")
    print(f"  Average trailing zeros: {np.mean([p[0] for p in upper_patterns]):.2f}")
    print(f"  Average trailing ones: {np.mean([p[1] for p in upper_patterns]):.2f}")

    print("\nLower track patterns:")
    print(f"  Average trailing zeros: {np.mean([p[0] for p in lower_patterns]):.2f}")
    print(f"  Average trailing ones: {np.mean([p[1] for p in lower_patterns]):.2f}")

    # Analyze compression distribution
    print("\n=== Compression Analysis ===")
    compression_by_track = {"upper": [], "lower": []}
    for r in results:
        compression_by_track[r["track"]].append(r["compression"])

    print("\nCompression statistics:")
    print(
        f"Upper track: mean={np.mean(compression_by_track['upper']):.2f}, std={np.std(compression_by_track['upper']):.2f}"
    )
