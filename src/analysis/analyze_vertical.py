import math
import numpy as np
import matplotlib.pyplot as plt


def analyze_vertical_structure(n):
    bits_before = len(format(n, "b"))
    next_n = 3 * n + 1
    tau = 0
    while next_n % 2 == 0:
        tau += 1
        next_n //= 2
    bits_after = len(format(next_n, "b"))
    return {
        "n": n,
        "bits_before": bits_before,
        "bits_after": bits_after,
        "tau": tau,
        "binary": format(n, "b"),
    }


# Analyze numbers up to 13 bits (8192)
results = [analyze_vertical_structure(n) for n in range(1, 8192, 2)]

# Group by input bit length
by_input_bits = {}
for r in results:
    bits = r["bits_before"]
    if bits not in by_input_bits:
        by_input_bits[bits] = []
    by_input_bits[bits].append(r)

print("=== Vertical Line Analysis ===")
for bits in sorted(by_input_bits.keys()):
    points = by_input_bits[bits]
    output_bits = [p["bits_after"] for p in points]
    taus = [p["tau"] for p in points]

    print(f"\nInput bits: {bits}")
    print(f"Number of points: {len(points)}")
    print(f"Output bits range: {min(output_bits)} to {max(output_bits)}")
    print(f"Tau values: {sorted(set(taus))}")

    if bits <= 4:  # Detailed analysis of early range
        print("\nDetailed early range analysis:")
        for p in points:
            print(
                f"  n={int(p['binary'], 2)}: τ={p['tau']}, out_bits={p['bits_after']}"
            )

# Create visualization focusing on vertical structure
plt.figure(figsize=(15, 10))

# Plot points with size proportional to frequency
for bits in sorted(by_input_bits.keys()):
    points = by_input_bits[bits]
    x = [p["bits_before"] for p in points]
    y = [p["bits_after"] for p in points]

    # Count frequency of each (x,y) combination
    from collections import Counter

    freq = Counter(zip(x, y))

    # Plot each point with size proportional to frequency
    for (px, py), count in freq.items():
        plt.scatter(px, py, s=count * 50, alpha=0.5, c="blue")

# Add the prediction line
max_bits = max(r["bits_before"] for r in results)
plt.plot([0, max_bits], [0, max_bits], "r--", label="y=x")

plt.xlabel("Input Bits")
plt.ylabel("Output Bits")
plt.title(
    "Vertical Structure Analysis of Bit Evolution\nPoint size indicates frequency"
)
plt.grid(True, alpha=0.3)

# Annotate early range points
for bits in range(1, 5):
    if bits in by_input_bits:
        for p in by_input_bits[bits]:
            plt.annotate(
                f"{int(p['binary'], 2)}\nτ={p['tau']}",
                (p["bits_before"], p["bits_after"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

plt.savefig("entropy/vertical_structure.png", dpi=300, bbox_inches="tight")
plt.close()

# Analyze vertical spacing
print("\n=== Vertical Spacing Analysis ===")
for bits in sorted(by_input_bits.keys()):
    if bits >= 5:  # Skip early range
        points = by_input_bits[bits]
        output_bits = sorted(set(p["bits_after"] for p in points))
        if len(output_bits) > 1:
            spacings = [
                output_bits[i + 1] - output_bits[i] for i in range(len(output_bits) - 1)
            ]
            print(f"\nBit length {bits}:")
            print(f"Number of distinct output lengths: {len(output_bits)}")
            print(f"Spacings between outputs: {spacings}")
