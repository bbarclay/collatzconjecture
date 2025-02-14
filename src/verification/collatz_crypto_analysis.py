"""
Cryptographic Analysis of the Collatz Function

This module provides tools to analyze and demonstrate how the Collatz function
exhibits properties similar to cryptographic hash functions, specifically:
1. Avalanche effect
2. One-way properties
3. Information loss
4. Round function behavior

The Collatz function is a natural cryptographic hash because it combines three operations:
1. ×3: Expansion phase (increases entropy and bit length)
2. +1: Bit mixing phase (creates avalanche effect)
3. ÷2^τ: Compression phase (information loss)

Key Insight: The Collatz function exhibits two fundamentally different behaviors:
1. Power of 2 Line: Pure collapse through division (predictable, reversible)
2. 3n+1 Churning: Cryptographic mixing operation (unpredictable, irreversible)

The cryptographic strength comes from the unpredictable interleaving of these operations,
where the 3n+1 operation acts as a "churning" mechanism that creates avalanche effects
and prevents prediction of when a number will hit the power of 2 line.
"""

from typing import Dict, List, Tuple, NamedTuple, Optional
from dataclasses import dataclass
import math
from collections import defaultdict


@dataclass
class BitTransformation:
    """Tracks bit-level changes during Collatz operations."""

    input_bits: str
    output_bits: str
    bits_changed: int
    positions_changed: List[int]
    entropy_change: float

    @classmethod
    def from_numbers(cls, input_num: int, output_num: int) -> "BitTransformation":
        input_bits = format(input_num, "b")
        output_bits = format(output_num, "b")
        max_len = max(len(input_bits), len(output_bits))
        input_padded = input_bits.zfill(max_len)
        output_padded = output_bits.zfill(max_len)

        positions = [i for i in range(max_len) if input_padded[i] != output_padded[i]]
        entropy_change = (
            math.log2(output_num) - math.log2(input_num) if input_num > 0 else 0
        )

        return cls(
            input_bits=input_bits,
            output_bits=output_bits,
            bits_changed=len(positions),
            positions_changed=positions,
            entropy_change=entropy_change,
        )


@dataclass
class CollatzRound:
    """Represents one round of Collatz transformation."""

    input_value: int
    is_odd: bool
    # For odd numbers:
    triple_value: Optional[int] = None
    plus_one_value: Optional[int] = None
    divisions: int = 0
    # Transformations:
    triple_transform: Optional[BitTransformation] = None
    plus_one_transform: Optional[BitTransformation] = None
    division_transform: Optional[BitTransformation] = None
    # Final result
    output_value: int = 0

    def analyze_transforms(self) -> None:
        """Analyze all transformations in this round."""
        if self.is_odd:
            self.triple_transform = BitTransformation.from_numbers(
                self.input_value, self.triple_value
            )
            self.plus_one_transform = BitTransformation.from_numbers(
                self.triple_value, self.plus_one_value
            )
            self.division_transform = BitTransformation.from_numbers(
                self.plus_one_value, self.output_value
            )
        else:
            self.division_transform = BitTransformation.from_numbers(
                self.input_value, self.output_value
            )


@dataclass
class ThreePhaseAnalysis:
    """Analyzes the three phases of a Collatz odd-step transformation."""

    input_value: int
    # Phase 1: ×3
    expansion_value: int
    expansion_bits: str
    bits_added: int
    # Phase 2: +1
    mixed_value: int
    mixed_bits: str
    bits_flipped: int
    # Phase 3: ÷2^τ
    tau_value: int
    output_value: int
    output_bits: str
    bits_removed: int
    # Overall effect
    net_entropy_change: float

    @classmethod
    def from_odd_number(cls, n: int) -> "ThreePhaseAnalysis":
        # Phase 1: ×3
        expansion = 3 * n
        exp_bits = format(expansion, "b")
        orig_bits = format(n, "b")
        bits_added = len(exp_bits) - len(orig_bits)

        # Phase 2: +1
        mixed = expansion + 1
        mixed_bits = format(mixed, "b")
        bits_flipped = sum(
            1
            for i in range(len(exp_bits))
            if exp_bits[-(i + 1) : -(i)] != mixed_bits[-(i + 1) : -(i)]
        )

        # Phase 3: ÷2^τ
        tau = 0
        output = mixed
        while output % 2 == 0:
            output //= 2
            tau += 1
        output_bits = format(output, "b")
        bits_removed = len(mixed_bits) - len(output_bits)

        # Calculate entropy change
        entropy_change = math.log2(output) - math.log2(n)

        return cls(
            input_value=n,
            expansion_value=expansion,
            expansion_bits=exp_bits,
            bits_added=bits_added,
            mixed_value=mixed,
            mixed_bits=mixed_bits,
            bits_flipped=bits_flipped,
            tau_value=tau,
            output_value=output,
            output_bits=output_bits,
            bits_removed=bits_removed,
            net_entropy_change=entropy_change,
        )


@dataclass
class PowerOf2Analysis:
    """Analyzes behavior relative to powers of 2."""

    value: int
    nearest_power_2: int
    distance_to_power_2: int
    pure_collapse: bool  # True if only divisions occur
    steps_to_collapse: int

    @classmethod
    def from_number(cls, n: int) -> "PowerOf2Analysis":
        # Find nearest power of 2
        log2_val = math.log2(n)
        floor_power = 2 ** int(log2_val)
        ceil_power = 2 ** math.ceil(log2_val)
        nearest_power = floor_power if n - floor_power < ceil_power - n else ceil_power

        # Check if it's on power of 2 line
        current = n
        steps = 0
        pure_collapse = True

        while current > 1 and pure_collapse:
            if current % 2 == 0:
                current //= 2
                steps += 1
            else:
                pure_collapse = False

        return cls(
            value=n,
            nearest_power_2=nearest_power,
            distance_to_power_2=abs(n - nearest_power),
            pure_collapse=pure_collapse,
            steps_to_collapse=steps if pure_collapse else 0,
        )


class CollatzCryptoAnalyzer:
    """Analyzes cryptographic properties of the Collatz function."""

    def __init__(self):
        self.round_history: Dict[int, List[CollatzRound]] = {}

    def analyze_number(self, n: int, max_rounds: int = 10) -> List[CollatzRound]:
        """Analyze the cryptographic properties of Collatz sequence for n."""
        rounds = []
        current = n

        for _ in range(max_rounds):
            if current <= 1:
                break

            round = self._analyze_single_round(current)
            rounds.append(round)
            current = round.output_value

        self.round_history[n] = rounds
        return rounds

    def _analyze_single_round(self, n: int) -> CollatzRound:
        """Analyze a single round of Collatz transformation."""
        if n % 2 == 0:
            round = CollatzRound(input_value=n, is_odd=False, output_value=n // 2)
        else:
            triple = 3 * n
            plus_one = triple + 1
            divisions = 0
            result = plus_one

            while result % 2 == 0:
                divisions += 1
                result //= 2

            round = CollatzRound(
                input_value=n,
                is_odd=True,
                triple_value=triple,
                plus_one_value=plus_one,
                divisions=divisions,
                output_value=result,
            )

        round.analyze_transforms()
        return round

    def analyze_avalanche(self, n1: int, n2: int) -> Dict:
        """Analyze avalanche effect between two numbers."""
        rounds1 = self.analyze_number(n1)
        rounds2 = self.analyze_number(n2)

        input_diff = BitTransformation.from_numbers(n1, n2)

        results = {"input_difference": input_diff, "round_differences": []}

        for i in range(min(len(rounds1), len(rounds2))):
            round_diff = BitTransformation.from_numbers(
                rounds1[i].output_value, rounds2[i].output_value
            )
            results["round_differences"].append(round_diff)

        return results

    def analyze_bit_influence(self, n: int) -> Dict[int, List[float]]:
        """Analyze how each input bit influences future bits."""
        influence = defaultdict(list)
        original_path = self.analyze_number(n)

        # Flip each bit and analyze changes
        for bit in range(int(math.log2(n)) + 1):
            flipped = n ^ (1 << bit)
            flipped_path = self.analyze_number(flipped)

            # Track differences through rounds
            for round_idx in range(min(len(original_path), len(flipped_path))):
                orig = original_path[round_idx].output_value
                flip = flipped_path[round_idx].output_value
                diff = BitTransformation.from_numbers(orig, flip)
                influence[bit].append(diff.bits_changed / len(diff.input_bits))

        return influence

    def analyze_three_phase_operation(self, n: int) -> ThreePhaseAnalysis:
        """Analyze the three-phase cryptographic operation for an odd number."""
        if n % 2 == 0:
            raise ValueError("Input must be odd")
        return ThreePhaseAnalysis.from_odd_number(n)

    def analyze_hash_properties(self, n: int, rounds: int = 5) -> Dict:
        """Analyze hash-like properties through multiple rounds."""
        results = {
            "input": n,
            "rounds": [],
            "total_entropy_change": 0,
            "total_bits_processed": 0,
        }

        current = n
        for _ in range(rounds):
            if current <= 1:
                break

            if current % 2 == 1:
                analysis = self.analyze_three_phase_operation(current)
                round_info = {
                    "type": "odd",
                    "analysis": analysis,
                    "entropy_change": analysis.net_entropy_change,
                    "bits_processed": (
                        analysis.bits_added
                        + analysis.bits_flipped
                        + analysis.bits_removed
                    ),
                }
                current = analysis.output_value
            else:
                # Even numbers just divide
                next_val = current // 2
                entropy_change = math.log2(next_val) - math.log2(current)
                round_info = {
                    "type": "even",
                    "input": current,
                    "output": next_val,
                    "entropy_change": entropy_change,
                    "bits_processed": 1,
                }
                current = next_val

            results["rounds"].append(round_info)
            results["total_entropy_change"] += round_info["entropy_change"]
            results["total_bits_processed"] += round_info["bits_processed"]

        return results

    def analyze_churning_vs_collapse(self, n: int, max_rounds: int = 10) -> Dict:
        """Analyze the interplay between 3n+1 churning and power of 2 collapse."""
        results = {
            "input": n,
            "rounds": [],
            "churning_phases": 0,
            "collapse_phases": 0,
            "total_bits_churned": 0,
        }

        current = n
        for round in range(max_rounds):
            if current <= 1:
                break

            power2_analysis = PowerOf2Analysis.from_number(current)

            if power2_analysis.pure_collapse:
                # We're on the power of 2 line - pure collapse
                phase_type = "collapse"
                next_val = current // 2
                bits_churned = 0
                results["collapse_phases"] += 1
            else:
                # We need 3n+1 operation - churning occurs
                phase_type = "churn"
                next_val = 3 * current + 1
                # Count bits churned through 3n+1
                bits_churned = bin(next_val ^ current).count("1")
                results["churning_phases"] += 1
                results["total_bits_churned"] += bits_churned

                # Divide out powers of 2
                while next_val % 2 == 0:
                    next_val //= 2

            round_info = {
                "value": current,
                "phase_type": phase_type,
                "bits_churned": bits_churned,
                "power2_analysis": power2_analysis,
                "next_value": next_val,
            }
            results["rounds"].append(round_info)
            current = next_val

        return results


def demonstrate_irreducibility():
    """Demonstrate why Collatz cannot be reduced to a single equation."""
    analyzer = CollatzCryptoAnalyzer()

    # Test cases demonstrating irreducibility
    test_cases = [
        (31, 32),  # Power of 2 boundary
        (63, 64),  # Power of 2 boundary
        (27, 31),  # Same bit length, different paths
        (15, 16),  # Power of 2 boundary
    ]

    results = []
    for n1, n2 in test_cases:
        avalanche = analyzer.analyze_avalanche(n1, n2)
        influence1 = analyzer.analyze_bit_influence(n1)
        influence2 = analyzer.analyze_bit_influence(n2)

        results.append(
            {
                "pair": (n1, n2),
                "avalanche": avalanche,
                "bit_influence": (influence1, influence2),
            }
        )

    return results


def demonstrate_hash_properties():
    """Demonstrate why Collatz acts like a cryptographic hash function."""
    analyzer = CollatzCryptoAnalyzer()

    # Test cases demonstrating hash-like behavior
    test_cases = [
        27,  # Famous long sequence
        31,  # Near power of 2
        41,  # Another interesting case
        63,  # Perfect chain number
    ]

    results = []
    for n in test_cases:
        hash_analysis = analyzer.analyze_hash_properties(n)
        results.append(hash_analysis)

    return results


def demonstrate_churning_effect():
    """Demonstrate the difference between power of 2 collapse and 3n+1 churning."""
    analyzer = CollatzCryptoAnalyzer()

    # Test cases showing different behaviors
    test_cases = [
        32,  # Pure power of 2 - collapse only
        31,  # Just below power of 2
        27,  # Famous long sequence
        255,  # 2^8 - 1, maximum churning before collapse
    ]

    results = []
    for n in test_cases:
        analysis = analyzer.analyze_churning_vs_collapse(n)
        results.append(analysis)

    return results


def analyze_predecessors(m: int, max_k: int = 20) -> List[Tuple[int, int]]:
    """
    Find and analyze all predecessors of an odd number m.

    Args:
        m: The target odd number
        max_k: Maximum power of 2 to check

    Returns:
        List of tuples (predecessor, k) where k is the power of 2
    """
    if m <= 0 or m % 2 == 0:
        raise ValueError("Input must be a positive odd integer")

    predecessors = []
    for k in range(1, max_k + 1):
        # Solve 3n + 1 = m * 2^k
        n = (m * (2**k) - 1) // 3
        if n > 0 and n % 2 == 1 and (3 * n + 1) // (2**k) == m:
            predecessors.append((n, k))
    return predecessors


def analyze_predecessor_pattern(m: int, max_k: int = 20) -> Dict:
    """
    Analyze the pattern of k values and spacing in predecessors.

    Args:
        m: The target odd number
        max_k: Maximum power of 2 to check

    Returns:
        Dictionary containing pattern analysis
    """
    preds = analyze_predecessors(m, max_k)
    if not preds:
        return {"has_pattern": False, "predecessors": 0}

    k_values = [k for _, k in preds]
    k_diffs = [k_values[i + 1] - k_values[i] for i in range(len(k_values) - 1)]

    # Check if k differences are constant
    constant_diff = len(set(k_diffs)) == 1

    # Analyze growth of predecessor values
    pred_values = [n for n, _ in preds]
    growth_ratios = [
        pred_values[i + 1] / pred_values[i] for i in range(len(pred_values) - 1)
    ]

    return {
        "has_pattern": constant_diff,
        "predecessors": len(preds),
        "k_values": k_values,
        "k_differences": k_diffs,
        "constant_difference": k_diffs[0] if constant_diff else None,
        "growth_ratios": growth_ratios,
        "average_growth": (
            sum(growth_ratios) / len(growth_ratios) if growth_ratios else None
        ),
    }


if __name__ == "__main__":
    # Original irreducibility demonstration
    results = demonstrate_irreducibility()

    print("Cryptographic Analysis of Collatz Function")
    print("=========================================")

    for result in results:
        n1, n2 = result["pair"]
        print(f"\nAnalyzing pair {n1} vs {n2}:")

        # Show avalanche effect
        av = result["avalanche"]
        print(f"Input difference: {av['input_difference'].bits_changed} bits")
        print("Round-by-round differences:")
        for i, diff in enumerate(av["round_differences"]):
            print(f"  Round {i+1}: {diff.bits_changed} bits changed")

        # Show bit influence
        inf1, inf2 = result["bit_influence"]
        print(f"\nBit influence analysis for {n1}:")
        for bit, influences in inf1.items():
            avg_influence = sum(influences) / len(influences)
            print(f"  Bit {bit} influences {avg_influence:.2%} of future bits")

        print(f"\nBit influence analysis for {n2}:")
        for bit, influences in inf2.items():
            avg_influence = sum(influences) / len(influences)
            print(f"  Bit {bit} influences {avg_influence:.2%} of future bits")

    # New hash property demonstration
    print("\nHash-Like Properties Analysis")
    print("============================")

    hash_results = demonstrate_hash_properties()
    for result in hash_results:
        print(f"\nAnalyzing number {result['input']}:")
        print(f"Total entropy change: {result['total_entropy_change']:.2f}")
        print(f"Total bits processed: {result['total_bits_processed']}")
        print("\nRound-by-round analysis:")

        for i, round in enumerate(result["rounds"]):
            print(f"\nRound {i+1}:")
            if round["type"] == "odd":
                analysis = round["analysis"]
                print("Three-phase transformation:")
                print(
                    f"1. ×3: {analysis.input_value} -> {analysis.expansion_value} "
                    f"(added {analysis.bits_added} bits)"
                )
                print(
                    f"2. +1: {analysis.expansion_value} -> {analysis.mixed_value} "
                    f"(flipped {analysis.bits_flipped} bits)"
                )
                print(
                    f"3. ÷2^{analysis.tau_value}: {analysis.mixed_value} -> "
                    f"{analysis.output_value} (removed {analysis.bits_removed} bits)"
                )
            else:
                print(f"Even step: {round['input']} -> {round['output']}")
            print(f"Entropy change this round: {round['entropy_change']:.2f}")

    # New churning vs collapse demonstration
    print("\nChurning vs Collapse Analysis")
    print("============================")

    churn_results = demonstrate_churning_effect()
    for result in churn_results:
        print(f"\nAnalyzing number {result['input']}:")
        print(f"Churning phases: {result['churning_phases']}")
        print(f"Collapse phases: {result['collapse_phases']}")
        print(f"Total bits churned: {result['total_bits_churned']}")
        print("\nRound-by-round analysis:")

        for i, round in enumerate(result["rounds"]):
            print(f"\nRound {i+1}:")
            print(f"Value: {round['value']}")
            print(f"Phase: {round['phase_type']}")
            if round["phase_type"] == "churn":
                print(f"Bits churned: {round['bits_churned']}")
            p2a = round["power2_analysis"]
            print(
                f"Distance to nearest power of 2 "
                f"({p2a.nearest_power_2}): {p2a.distance_to_power_2}"
            )

    # Analyze predecessor patterns
    print("\nPredecessor Pattern Analysis")
    print("===========================")

    test_numbers = [
        462247253,  # Our special number with evenly spaced k values
        27,  # Small number
        255,  # 2^8 - 1
        2**20 - 1,  # Another power of 2 minus 1
        999999999,  # Large number for comparison
    ]

    for n in test_numbers:
        print(f"\nAnalyzing patterns for {n}:")
        pattern = analyze_predecessor_pattern(n)
        if pattern["has_pattern"]:
            print(f"Found {pattern['predecessors']} predecessors with regular pattern:")
            print(f"k values: {pattern['k_values']}")
            print(
                f"Constant difference between k values: {pattern['constant_difference']}"
            )
            print(
                f"Average growth ratio between predecessors: {pattern['average_growth']:.2f}"
            )
        else:
            print(
                f"No regular pattern found. Has {pattern['predecessors']} predecessors."
            )
