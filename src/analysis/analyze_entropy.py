import math
import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_bit_pattern(n):
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
        "next_n": next_n,
    }


# Create entropy directory if it doesn't exist
if not os.path.exists("entropy"):
    os.makedirs("entropy")

# Analyze a much larger range
results = [analyze_bit_pattern(n) for n in range(1, 10000, 2)]

# Create scatter plot with more information
plt.figure(figsize=(12, 8))

# Plot points with different colors for different residue classes
colors = ["blue", "green", "red"]
markers = ["o", "s", "^"]
labels = ["mod 3 ≡ 0", "mod 3 ≡ 1", "mod 3 ≡ 2"]

for residue in [0, 1, 2]:
    residue_points = [r for r in results if r["residue"] == residue]
    x = [r["bits_before"] for r in residue_points]
    y = [r["bits_after"] for r in residue_points]
    plt.scatter(
        x,
        y,
        c=colors[residue],
        marker=markers[residue],
        alpha=0.5,
        label=labels[residue],
    )

# Add the y=x line
max_bits = max(
    max(r["bits_before"] for r in results), max(r["bits_after"] for r in results)
)
plt.plot(
    [0, max_bits], [0, max_bits], "r--", alpha=0.5, label="Perfect Prediction (y=x)"
)

# Add predicted line (y = x + floor(log2(3)))
x_range = np.array([0, max_bits])
y_pred = x_range + math.floor(math.log2(3))
plt.plot(x_range, y_pred, "g--", alpha=0.5, label="Theoretical Upper Track")

plt.xlabel("Input Bits")
plt.ylabel("Output Bits")
plt.title("Bit Length Evolution in Collatz Steps\nColored by Residue Class mod 3")
plt.legend()
plt.grid(True, alpha=0.3)

# Add annotations for interesting points
for r in results[:20]:  # Annotate first few points
    if r["bits_before"] < 5:  # Only annotate small numbers
        plt.annotate(
            f"{r['n']}\nτ={r['tau']}",
            (r["bits_before"], r["bits_after"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

plt.savefig("entropy/bit_evolution_detailed.png", dpi=300, bbox_inches="tight")
plt.close()

# Create histogram of compression amounts by track
plt.figure(figsize=(12, 6))
compression_upper = [r["compression"] for r in results if r["track"] == "upper"]
compression_lower = [r["compression"] for r in results if r["track"] == "lower"]

plt.hist(
    [compression_upper, compression_lower],
    label=["Upper Track", "Lower Track"],
    bins=20,
    alpha=0.7,
)
plt.xlabel("Compression Amount (bits)")
plt.ylabel("Frequency")
plt.title("Distribution of Compression by Track")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("entropy/compression_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# Print detailed statistics
print("=== Extended Analysis ===")
print(f"\nTotal numbers analyzed: {len(results)}")

for track in ["upper", "lower"]:
    track_results = [r for r in results if r["track"] == track]
    print(f"\n{track.upper()} TRACK:")
    print(f"Count: {len(track_results)}")
    print(f"Average τ: {np.mean([r['tau'] for r in track_results]):.2f}")
    print(
        f"Average compression: {np.mean([r['compression'] for r in track_results]):.2f}"
    )

    # Analyze by residue class
    for residue in [0, 1, 2]:
        residue_results = [r for r in track_results if r["residue"] == residue]
        count = len(residue_results)
        if count > 0:
            print(f"\nMod 3 ≡ {residue}:")
            print(f"  Count: {count}")
            print(f"  Average τ: {np.mean([r['tau'] for r in residue_results]):.2f}")
            print(
                f"  Average compression: {np.mean([r['compression'] for r in residue_results]):.2f}"
            )


# Analyze gaps between tracks
def analyze_track_gaps():
    gaps = []
    for bits in range(2, max(r["bits_before"] for r in results)):
        upper_points = [
            r for r in results if r["bits_before"] == bits and r["track"] == "upper"
        ]
        lower_points = [
            r for r in results if r["bits_before"] == bits and r["track"] == "lower"
        ]
        if upper_points and lower_points:
            avg_upper = np.mean([r["bits_after"] for r in upper_points])
            avg_lower = np.mean([r["bits_after"] for r in lower_points])
            gaps.append(avg_upper - avg_lower)
    return np.mean(gaps) if gaps else 0


print("\n=== Track Gap Analysis ===")
print(f"Average gap between tracks: {analyze_track_gaps():.2f} bits")
