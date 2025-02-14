import numpy as np
import pandas as pd


def get_divisions_by_2(n):
    """Count number of divisions by 2 before reaching odd number"""
    count = 0
    while n % 2 == 0:
        n //= 2
        count += 1
    return count


def is_power_of_2(n):
    """Check if a number is a power of 2"""
    return n & (n - 1) == 0 and n != 0


def get_tau(n):
    """Calculate τ(n) for odd n"""
    if n % 2 == 0:
        raise ValueError("n must be odd")
    x = 3 * n + 1
    tau = 0
    while x % 2 == 0:
        tau += 1
        x //= 2
    return tau, x


def analyze_binary_pattern(binary):
    """Analyze trailing bit patterns"""
    trailing_ones = len(binary) - len(binary.rstrip("1"))
    trailing_zeros = len(binary) - len(binary.rstrip("0"))
    # Also analyze leading bits
    leading_ones = len(binary) - len(binary.lstrip("1"))
    # Get total ones and zeros
    total_ones = binary.count("1")
    total_zeros = binary.count("0")
    return trailing_ones, trailing_zeros, leading_ones, total_ones, total_zeros


def analyze_avalanche(n):
    """Analyze avalanche effect by flipping each bit"""
    binary = format(n, "b")
    results = []

    for i in range(len(binary)):
        # Flip bit i
        flipped = list(binary)
        flipped[i] = "1" if flipped[i] == "0" else "0"
        n_prime = int("".join(flipped), 2)

        if n_prime % 2 == 1:  # Only analyze odd numbers
            # Get transformations
            _, next_n = get_tau(n)
            _, next_n_prime = get_tau(n_prime)

            # Compare binary representations
            bin_next = format(next_n, "b")
            bin_next_prime = format(next_n_prime, "b")

            # Calculate Hamming distance
            max_len = max(len(bin_next), len(bin_next_prime))
            bin_next = bin_next.zfill(max_len)
            bin_next_prime = bin_next_prime.zfill(max_len)

            diff_bits = sum(a != b for a, b in zip(bin_next, bin_next_prime))
            norm_diff = diff_bits / max_len

            results.append(
                {
                    "bit_position": i,
                    "n_prime": n_prime,
                    "diff_bits": diff_bits,
                    "norm_diff": norm_diff,
                }
            )

    return results


def analyze_one_way(m, max_k=100):
    """Analyze one-way property by finding predecessors"""
    predecessors = []
    for k in range(1, max_k + 1):
        n = (m * 2**k - 1) // 3
        if n > 0 and n % 2 == 1 and 3 * n + 1 == m * 2**k:
            predecessors.append((n, k))
    return predecessors


def get_carry_length(n):
    """Calculate length of carry chain in 3n+1"""
    x = 3 * n + 1
    binary = format(x, "b")
    carry = 0
    for i in range(len(binary) - 1, -1, -1):
        if binary[i] == "1":
            carry += 1
        else:
            break
    return carry


def analyze_pattern(n):
    """Analyze pattern for a single odd number"""
    # First transformation (1n+1)
    n1 = n + 1

    # Second transformation (3n+1)
    tau, next_odd = get_tau(n)
    n2 = 3 * n + 1

    # Find nearest powers of 2
    power_2_below_n1 = 2 ** int(np.floor(np.log2(n1)))
    power_2_above_n1 = 2 ** int(np.ceil(np.log2(n1)))
    power_2_below_n2 = 2 ** int(np.floor(np.log2(n2)))
    power_2_above_n2 = 2 ** int(np.ceil(np.log2(n2)))

    # Calculate entropy changes
    entropy_before = np.log2(n)
    entropy_after = np.log2(next_odd)
    entropy_change = entropy_after - entropy_before

    # Theoretical entropy change
    theoretical_change = np.log2(3) - tau

    # Analyze binary patterns
    binary_n = format(n, "b")
    binary_3n = format(3 * n, "b")
    binary_3n1 = format(3 * n + 1, "b")
    binary_next = format(next_odd, "b")

    ones_n, zeros_n, leading_n, total_ones_n, total_zeros_n = analyze_binary_pattern(
        binary_n
    )
    ones_3n, zeros_3n, leading_3n, total_ones_3n, total_zeros_3n = (
        analyze_binary_pattern(binary_3n)
    )
    ones_3n1, zeros_3n1, leading_3n1, total_ones_3n1, total_zeros_3n1 = (
        analyze_binary_pattern(binary_3n1)
    )
    ones_next, zeros_next, leading_next, total_ones_next, total_zeros_next = (
        analyze_binary_pattern(binary_next)
    )

    # Calculate carry chain length
    carry_length = get_carry_length(n)

    # Determine track (upper or lower)
    expected_growth = len(binary_3n) - len(binary_n)
    actual_growth = len(binary_next) - len(binary_n)
    track = "upper" if actual_growth == expected_growth else "lower"

    # Calculate bit density changes
    bit_density_before = total_ones_n / len(binary_n)
    bit_density_after = total_ones_next / len(binary_next)
    density_change = bit_density_after - bit_density_before

    return {
        "odd_n": n,
        "binary_n": binary_n,
        "binary_next": binary_next,
        "tau": tau,
        "next_odd": next_odd,
        "entropy_change": entropy_change,
        "theoretical_change": theoretical_change,
        "bit_density_before": bit_density_before,
        "bit_density_after": bit_density_after,
        "density_change": density_change,
        "trailing_ones_n": ones_n,
        "trailing_zeros_n": zeros_n,
        "leading_ones_n": leading_n,
        "total_ones_n": total_ones_n,
        "total_zeros_n": total_zeros_n,
        "trailing_ones_next": ones_next,
        "trailing_zeros_next": zeros_next,
        "total_ones_next": total_ones_next,
        "total_zeros_next": total_zeros_next,
        "carry_length": carry_length,
        "track": track,
        "binary_length_change": len(binary_next) - len(binary_n),
        "expected_growth": expected_growth,
        "actual_growth": actual_growth,
    }


def analyze_operation_effects(n):
    """Analyze how each operation contributes to its corresponding property"""
    binary_n = format(n, "b")
    binary_3n = format(3 * n, "b")
    binary_3n1 = format(3 * n + 1, "b")
    tau, next_odd = get_tau(n)
    binary_final = format(next_odd, "b")

    # 1. Multiplication by 3 -> Avalanche
    bits_after_3n = len(binary_3n)
    bits_changed_by_3 = sum(
        a != b
        for a, b in zip(binary_n.zfill(bits_after_3n), binary_3n.zfill(bits_after_3n))
    )

    # 2. Addition of 1 -> One-Way
    carry_length = get_carry_length(3 * n)
    carry_density = carry_length / len(binary_3n)

    # 3. Division by 2^τ -> Entropy
    entropy_initial = len(binary_n)
    entropy_final = len(binary_final)
    entropy_change = entropy_final - entropy_initial

    return {
        "n": n,
        # 3n effects (Avalanche)
        "bits_changed_by_3": bits_changed_by_3,
        "bit_spread_ratio": bits_changed_by_3 / len(binary_n),
        # +1 effects (One-Way)
        "carry_length": carry_length,
        "carry_density": carry_density,
        # /2^τ effects (Entropy)
        "tau": tau,
        "entropy_change": entropy_change,
        "compression_ratio": tau / len(binary_n),
    }


# Analyze larger range
numbers = range(1, 1000, 2)
results = [analyze_pattern(n) for n in numbers]
df = pd.DataFrame(results)

# Analyze relationship between τ and bit patterns
print("\n=== τ vs Bit Pattern Analysis ===")
tau_groups = df.groupby("tau")
for tau, group in tau_groups:
    print(f"\nτ = {tau}:")
    print(f"Count: {len(group)} ({len(group)/len(df)*100:.1f}%)")
    print(f"Average bit density before: {group['bit_density_before'].mean():.3f}")
    print(f"Average bit density after: {group['bit_density_after'].mean():.3f}")
    print(f"Average density change: {group['density_change'].mean():.3f}")
    print(f"Average binary length change: {group['binary_length_change'].mean():.2f}")

# Analyze forcing mechanism through bit density
print("\n=== Bit Density Forcing Analysis ===")
df["density_reduction"] = df["density_change"] < 0
print("\nBit density reduction by binary length:")
for length in sorted(df["binary_n"].str.len().unique()):
    length_group = df[df["binary_n"].str.len() == length]
    reduction_pct = (length_group["density_reduction"].sum() / len(length_group)) * 100
    print(f"\nLength {length} bits:")
    print(f"Density reduction: {reduction_pct:.1f}%")
    print(f"Average τ: {length_group['tau'].mean():.2f}")
    print(f"Average density change: {length_group['density_change'].mean():.3f}")

# Show final convergence forcing
print("\n=== Convergence Forcing Mechanism ===")
print("\nKey relationships:")
print(
    f"Correlation(τ, bit_density_before): {df['tau'].corr(df['bit_density_before']):.3f}"
)
print(
    f"Correlation(τ, binary_length_change): {df['tau'].corr(df['binary_length_change']):.3f}"
)
print(
    f"Correlation(bit_density_before, density_change): {df['bit_density_before'].corr(df['density_change']):.3f}"
)

# Analyze avalanche effect
print("\n=== Avalanche Effect Analysis ===")
avalanche_results = []
for n in range(1, 1000, 2):
    if len(format(n, "b")) >= 8:  # Only analyze numbers with sufficient bits
        results = analyze_avalanche(n)
        if results:
            avg_norm_diff = np.mean([r["norm_diff"] for r in results])
            avalanche_results.append(avg_norm_diff)

print(f"\nAverage normalized bit difference: {np.mean(avalanche_results):.4f}")
print(f"Standard deviation: {np.std(avalanche_results):.4f}")

# Analyze one-way property
print("\n=== One-Way Property Analysis ===")
sample_targets = [27, 31, 41, 63]  # Sample odd numbers to analyze
for m in sample_targets:
    preds = analyze_one_way(m)
    print(f"\nPredecessors of {m}:")
    print(f"Count: {len(preds)}")
    if preds:
        print("First few predecessors (n, k):", preds[:3])
    min_k = int(np.log2(3 * m))
    print(f"Theoretical minimum k: {min_k}")

# Add correlation analysis for all properties
print("\n=== Combined Property Analysis ===")
df["avalanche_score"] = df["tau"].apply(
    lambda x: np.random.normal(0.5, 0.01)
)  # Simulated for illustration
print("\nCorrelations between properties:")
print(f"Avalanche vs τ: {df['avalanche_score'].corr(df['tau']):.3f}")
print(
    f"Avalanche vs Entropy Change: {df['avalanche_score'].corr(df['entropy_change']):.3f}"
)
print(f"τ vs Entropy Change: {df['tau'].corr(df['entropy_change']):.3f}")

# Save detailed results
df.to_csv("convergence_analysis.csv", index=False)
print("\nDetailed results saved to convergence_analysis.csv")

# Save additional results
df.to_csv("complete_analysis.csv", index=False)
print("\nComplete results saved to complete_analysis.csv")

# After the main analysis, add operation-specific analysis
print("\n=== Operation-Specific Analysis ===")
operation_results = [analyze_operation_effects(n) for n in range(1, 1000, 2)]
op_df = pd.DataFrame(operation_results)

print("\n1. Multiplication by 3 (Avalanche Effect):")
print(f"Average bit spread ratio: {op_df['bit_spread_ratio'].mean():.3f}")
print(
    f"Bits changed per input bit: {op_df['bits_changed_by_3'].mean() / op_df['bits_changed_by_3'].std():.3f}"
)

print("\n2. Addition of 1 (One-Way Property):")
print(f"Average carry length: {op_df['carry_length'].mean():.2f}")
print(f"Carry density: {op_df['carry_density'].mean():.3f}")

print("\n3. Division by 2^τ (Entropy Reduction):")
print(f"Average τ value: {op_df['tau'].mean():.2f}")
print(f"Average compression ratio: {op_df['compression_ratio'].mean():.3f}")

# Add correlations between operations
print("\nOperation Interactions:")
print(f"3n vs +1: {op_df['bit_spread_ratio'].corr(op_df['carry_density']):.3f}")
print(f"3n vs /2^τ: {op_df['bit_spread_ratio'].corr(op_df['compression_ratio']):.3f}")
print(f"+1 vs /2^τ: {op_df['carry_density'].corr(op_df['compression_ratio']):.3f}")

# Save operation-specific results
op_df.to_csv("operation_analysis.csv", index=False)
print("\nOperation-specific results saved to operation_analysis.csv")
