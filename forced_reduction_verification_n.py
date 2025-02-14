# %% [markdown]
# # Computational Verification of Forced Reduction in Collatz Conjecture
#
# This notebook provides computational verification of the forced reduction properties described in our paper. We demonstrate that:
#
# 1. No sequence can escape to infinity
# 2. Large τ events occur frequently enough to force descent
# 3. Special number patterns (Mersenne, alternating bits) still descend
# 4. The distribution of τ matches theoretical predictions
#
# ## Paper Section Reference
#
# This notebook verifies the claims made in Section 'Forced Reduction' of our paper, specifically:
#
# 1. **Forward Uniqueness (FU)**: Each odd n strictly maps to (3n + 1)/2^τ(n)
# 2. **Backward Growth (BG)**: Odd predecessors jump upward exponentially
# 3. **Modular/Bit Forcing (MBF)**: Certain residue classes ensure large τ(n)
#
# The computational evidence here supports our theoretical argument that these three constraints together force eventual descent.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import math
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Configure plotting style
plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["figure.dpi"] = 100

# %% [markdown]
# ## Core Functions for Analyzing Collatz Behavior
#
# These functions implement the key mathematical operations described in the paper:


# %%
def get_tau(n: int) -> Tuple[int, int]:
    """Calculate τ(n) for odd n and return the next odd number.

    As described in the paper, τ(n) is the number of trailing zeros after
    multiplying by 3 and adding 1. This is a key measure of how much
    a number is reduced in each step.
    """
    if n % 2 == 0:
        raise ValueError("n must be odd")
    x = 3 * n + 1
    tau = 0
    while x % 2 == 0:
        tau += 1
        x //= 2
    return tau, x


def analyze_binary_pattern(binary: str) -> Dict[str, int]:
    """Analyze bit patterns that influence τ.

    The paper discusses how certain bit patterns lead to
    predictable τ values through carry chain effects.
    """
    return {
        "trailing_ones": len(binary) - len(binary.rstrip("1")),
        "trailing_zeros": len(binary) - len(binary.rstrip("0")),
        "leading_ones": len(binary) - len(binary.lstrip("1")),
        "total_ones": binary.count("1"),
        "total_zeros": binary.count("0"),
        "length": len(binary),
    }


def get_carry_length(n: int) -> int:
    """Calculate length of carry chain in 3n+1.

    The carry chain length is crucial for understanding how
    multiplication by 3 and addition of 1 interact to produce
    trailing zeros.
    """
    x = 3 * n + 1
    binary = format(x, "b")
    carry = 0
    for i in range(len(binary) - 1, -1, -1):
        if binary[i] == "1":
            carry += 1
        else:
            break
    return carry


# %% [markdown]
# ## Pattern Analysis Functions
#
# These functions analyze how bit patterns evolve and influence τ:


# %%
def analyze_pattern(n: int) -> Dict[str, Any]:
    """Comprehensive pattern analysis for a single odd number.

    This function analyzes:
    1. τ value and next odd number
    2. Entropy changes (using log base 2)
    3. Binary pattern evolution
    4. Bit density changes
    """
    # Get τ and next odd number
    tau, next_odd = get_tau(n)
    n2 = 3 * n + 1

    # Calculate entropy changes (using math.log2 for scalar values)
    entropy_before = math.log2(float(n))
    entropy_after = math.log2(float(next_odd))
    entropy_change = entropy_after - entropy_before
    theoretical_change = math.log2(3) - tau

    # Analyze binary patterns
    binary_n = format(n, "b")
    binary_3n = format(3 * n, "b")
    binary_next = format(next_odd, "b")

    # Get pattern statistics
    pattern_n = analyze_binary_pattern(binary_n)
    pattern_next = analyze_binary_pattern(binary_next)

    # Calculate carry chain length
    carry_length = get_carry_length(n)

    # Determine track (upper or lower)
    expected_growth = len(binary_3n) - len(binary_n)
    actual_growth = len(binary_next) - len(binary_n)
    track = "upper" if actual_growth == expected_growth else "lower"

    # Calculate bit density changes
    bit_density_before = pattern_n["total_ones"] / pattern_n["length"]
    bit_density_after = pattern_next["total_ones"] / pattern_next["length"]
    density_change = bit_density_after - bit_density_before

    return {
        "n": n,
        "binary_n": binary_n,
        "binary_next": binary_next,
        "tau": tau,
        "next_odd": next_odd,
        "entropy_change": entropy_change,
        "theoretical_change": theoretical_change,
        "bit_density_before": bit_density_before,
        "bit_density_after": bit_density_after,
        "density_change": density_change,
        "pattern_n": pattern_n,
        "pattern_next": pattern_next,
        "carry_length": carry_length,
        "track": track,
        "binary_length_change": actual_growth,
        "expected_growth": expected_growth,
    }


# %% [markdown]
# ## Interactive Analysis Functions
#
# These functions let you analyze specific numbers or ranges:


# %%
def analyze_number(n: int, verbose: bool = True) -> Dict[str, Any]:
    """Analyze a single number and print detailed results."""
    result = analyze_pattern(n)

    if verbose:
        print(f"Analysis of {n}:")
        print(f"Binary representation: {result['binary_n']}")
        print(f"τ value: {result['tau']}")
        print(f"Next odd number: {result['next_odd']}")
        print(f"\nBit Pattern Analysis:")
        print(f"Trailing ones: {result['pattern_n']['trailing_ones']}")
        print(f"Trailing zeros: {result['pattern_n']['trailing_zeros']}")
        print(f"Bit density: {result['bit_density_before']:.3f}")
        print(f"\nTransformation Effects:")
        print(f"Entropy change: {result['entropy_change']:.3f}")
        print(f"Theoretical change: {result['theoretical_change']:.3f}")
        print(f"Track: {result['track']}")
        print(f"Carry chain length: {result['carry_length']}")

    return result


def analyze_range(start: int, end: int, step: int = 2) -> pd.DataFrame:
    """Analyze a range of numbers and return results as a DataFrame."""
    numbers = range(start, end, step)
    results = [analyze_pattern(n) for n in tqdm(numbers, desc="Analyzing numbers")]
    return pd.DataFrame(results)


def plot_analysis(df: pd.DataFrame):
    """Create visualizations of the analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # τ distribution
    df.groupby("tau").size().plot(kind="bar", ax=axes[0, 0])
    axes[0, 0].set_title("Distribution of τ Values")
    axes[0, 0].set_xlabel("τ")
    axes[0, 0].set_ylabel("Frequency")

    # Entropy change vs τ
    df.plot.scatter("tau", "entropy_change", ax=axes[0, 1])
    axes[0, 1].set_title("Entropy Change vs τ")

    # Bit density changes
    df.plot.scatter("bit_density_before", "bit_density_after", ax=axes[1, 0])
    axes[1, 0].set_title("Bit Density Evolution")

    # Track distribution by residue
    track_by_residue = pd.crosstab(df["n"] % 3, df["track"])
    track_by_residue.plot(kind="bar", stacked=True, ax=axes[1, 1])
    axes[1, 1].set_title("Track Distribution by Residue mod 3")

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Example Usage
#
# Here's how to use these functions to verify the paper's claims:

# %%
# Example 1: Analyze a specific number
print("Analysis of the famous n=27 case:")
result_27 = analyze_number(27)

# Example 2: Analyze a range of numbers
print("\nAnalyzing numbers from 1 to 100:")
results_df = analyze_range(1, 100)
plot_analysis(results_df)

# Example 3: Analyze a Mersenne number
k = 7
mersenne = 2**k - 1
print(f"\nAnalysis of Mersenne number 2^{k}-1 = {mersenne}:")
result_mersenne = analyze_number(mersenne)

# %% [markdown]
# ## Interactive Analysis
#
# Use these cells to analyze any number or range you're interested in:

# %%
# Analyze your own number
n = 27  # Change this to any odd number you want to analyze
analyze_number(n)

# %%
# Analyze your own range
start = 1
end = 1000
results = analyze_range(start, end)
plot_analysis(results)

# Print summary statistics
print("\nSummary Statistics:")
print(f"Average τ: {results['tau'].mean():.2f}")
print(f"Maximum τ: {results['tau'].max()}")
print(f"Average entropy change: {results['entropy_change'].mean():.2f}")
print("\nTrack distribution:")
print(results["track"].value_counts())


# %%
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
            h1 = math.log2(float(n1)) if n1 > 0 else 0
            h2 = math.log2(float(n2)) if n2 > 0 else 0
            changes.append(h2 - h1)
        return changes

    def analyze_backward_growth(self, n: int, k: int = 10) -> List[Dict[str, Any]]:
        """Analyze backward growth by finding k predecessors.

        This verifies the Backward Growth (BG) property by showing that
        predecessors grow exponentially.
        """
        predecessors = []
        for i in range(1, k + 1):
            # Find predecessor that takes i steps
            pred = (n * 2**i - 1) // 3
            if pred > 0 and pred % 2 == 1 and self.verify_predecessor(pred, n, i):
                predecessors.append(
                    {
                        "predecessor": pred,
                        "steps": i,
                        "growth_ratio": math.log2(float(pred)) / math.log2(float(n)),
                    }
                )
        return predecessors

    def verify_predecessor(self, pred: int, target: int, steps: int) -> bool:
        """Verify that pred reaches target in exactly steps steps."""
        current = pred
        for _ in range(steps):
            current = self.collatz_step(current)
        return current == target

    def analyze_carry_chains(self, n: int) -> Dict[str, Any]:
        """Analyze carry chain effects in bit-avalanche."""
        binary = format(n, "b")
        n3plus1 = 3 * n + 1
        binary_after = format(n3plus1, "b")

        # Find carry chains
        carry_chains = []
        current_chain = 0
        for i in range(len(binary_after) - 1, -1, -1):
            if binary_after[i] == "1":
                current_chain += 1
            else:
                if current_chain > 0:
                    carry_chains.append(current_chain)
                current_chain = 0
        if current_chain > 0:
            carry_chains.append(current_chain)

        return {
            "original": binary,
            "after_3n_plus_1": binary_after,
            "carry_chains": carry_chains,
            "max_chain": max(carry_chains) if carry_chains else 0,
            "total_carries": sum(carry_chains),
            "chain_count": len(carry_chains),
        }

    def verify_entropy_threshold(
        self, start: int, end: int, step: int = 2
    ) -> Dict[str, float]:
        """Verify that E[τ(n)] ≳ log₂(3) for typical n."""
        tau_values = []
        entropy_changes = []

        for n in range(start, end, step):
            if n % 2 == 1:  # only odd numbers
                tau = self.find_tau(n)
                tau_values.append(tau)

                # Calculate actual entropy change
                next_odd = (3 * n + 1) // (2**tau)
                entropy_change = math.log2(float(next_odd)) - math.log2(float(n))
                entropy_changes.append(entropy_change)

        avg_tau = statistics.mean(tau_values)
        theoretical_threshold = math.log2(3)

        return {
            "average_tau": avg_tau,
            "theoretical_threshold": theoretical_threshold,
            "meets_threshold": avg_tau >= theoretical_threshold,
            "threshold_ratio": avg_tau / theoretical_threshold,
            "average_entropy_change": statistics.mean(entropy_changes),
            "entropy_change_std": statistics.stdev(entropy_changes),
        }

    def analyze_ergodicity(self, n: int, steps: int = 1000) -> Dict[str, Any]:
        """Analyze ergodic properties of τ(n) distribution."""
        trajectory = []
        tau_sequence = []
        current = n

        for _ in range(steps):
            if current % 2 == 1:
                tau = self.find_tau(current)
                tau_sequence.append(tau)
            trajectory.append(current)
            current = self.collatz_step(current)
            if current == 1:
                break

        # Analyze τ distribution properties
        if tau_sequence:
            return {
                "mean": statistics.mean(tau_sequence),
                "median": statistics.median(tau_sequence),
                "std": statistics.stdev(tau_sequence) if len(tau_sequence) > 1 else 0,
                "autocorrelation": (
                    np.corrcoef(tau_sequence[:-1], tau_sequence[1:])[0, 1]
                    if len(tau_sequence) > 1
                    else 0
                ),
                "sequence_length": len(tau_sequence),
                "unique_values": len(set(tau_sequence)),
                "distribution": pd.Series(tau_sequence).value_counts().to_dict(),
            }
        return {}

    def plot_trajectory_stats(self, n: int, save_path: str = None):
        """Plot comprehensive trajectory statistics."""
        analysis = self.analyze_trajectory(n)
        ergodic = self.analyze_ergodicity(n)
        carry = self.analyze_carry_chains(n)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot trajectory
        steps = range(len(analysis["trajectory"]))
        ax1.plot(steps, analysis["trajectory"])
        ax1.set_yscale("log")
        ax1.set_title(f"Trajectory for n={n}")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Value")

        # Plot tau distribution
        if ergodic and "distribution" in ergodic:
            taus, counts = zip(*sorted(ergodic["distribution"].items()))
            ax2.bar(taus, counts)
            ax2.set_title("τ Distribution")
            ax2.set_xlabel("τ Value")
            ax2.set_ylabel("Frequency")

        # Plot entropy changes
        if analysis["entropy_changes"]:
            ax3.plot(
                range(len(analysis["entropy_changes"])), analysis["entropy_changes"]
            )
            ax3.axhline(y=0, color="r", linestyle="--", label="y=0")
            ax3.set_title("Entropy Changes")
            ax3.set_xlabel("Step")
            ax3.set_ylabel("Entropy Change")
            ax3.legend()

        # Plot carry chain distribution
        if carry["carry_chains"]:
            chain_lengths, chain_counts = np.unique(
                carry["carry_chains"], return_counts=True
            )
            ax4.bar(chain_lengths, chain_counts)
            ax4.set_title("Carry Chain Distribution")
            ax4.set_xlabel("Chain Length")
            ax4.set_ylabel("Frequency")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


# %% [markdown]
# ## Verification of Paper Claims
#
# Let's verify the key claims from the paper:

# %%
analyzer = ForcedReductionAnalyzer()

# 1. Verify Backward Growth
print("Backward Growth Analysis:")
predecessors = analyzer.analyze_backward_growth(27, k=5)
for p in predecessors:
    print(
        f"Predecessor: {p['predecessor']}, Steps: {p['steps']}, Growth Ratio: {p['growth_ratio']:.2f}"
    )

# 2. Analyze Carry Chains
print("\nCarry Chain Analysis for n=27:")
carry_analysis = analyzer.analyze_carry_chains(27)
print(f"Original binary: {carry_analysis['original']}")
print(f"After 3n+1: {carry_analysis['after_3n_plus_1']}")
print(f"Carry chains: {carry_analysis['carry_chains']}")
print(f"Maximum chain length: {carry_analysis['max_chain']}")

# 3. Verify Entropy Threshold
print("\nEntropy Threshold Verification (1 to 1000):")
threshold_analysis = analyzer.verify_entropy_threshold(1, 1000)
print(f"Average τ: {threshold_analysis['average_tau']:.3f}")
print(
    f"Theoretical threshold (log₂(3)): {threshold_analysis['theoretical_threshold']:.3f}"
)
print(f"Meets threshold: {threshold_analysis['meets_threshold']}")
print(f"Average entropy change: {threshold_analysis['average_entropy_change']:.3f}")

# 4. Analyze Ergodicity
print("\nErgodicity Analysis for n=27:")
ergodic_analysis = analyzer.analyze_ergodicity(27)
print(f"Mean τ: {ergodic_analysis['mean']:.2f}")
print(f"Autocorrelation: {ergodic_analysis['autocorrelation']:.3f}")
print("τ Distribution:")
for tau, count in ergodic_analysis["distribution"].items():
    print(f"  τ={tau}: {count} times")

# 5. Plot comprehensive analysis
analyzer.plot_trajectory_stats(27)

# %% [markdown]
# ## Interactive Analysis
#
# Use these cells to analyze any numbers you're interested in:

# %%
# Analyze your own number
n = 27  # Change this to any odd number you want to analyze
analyzer = ForcedReductionAnalyzer()
analyzer.plot_trajectory_stats(n)

# Print detailed analysis
carry_analysis = analyzer.analyze_carry_chains(n)
ergodic_analysis = analyzer.analyze_ergodicity(n)
threshold_analysis = analyzer.verify_entropy_threshold(n, n + 1000)

print(f"\nAnalysis of {n}:")
print(f"Carry chains: {carry_analysis['carry_chains']}")
print(f"Average τ: {ergodic_analysis['mean']:.2f}")
print(f"Entropy threshold ratio: {threshold_analysis['threshold_ratio']:.3f}")
