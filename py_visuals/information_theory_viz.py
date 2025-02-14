import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def plot_entropy_reduction(max_n=100, output_path="figures/entropy_reduction.svg"):
    """Create a visualization of entropy reduction in Collatz steps."""

    def entropy_change(n):
        if n % 2 == 0:
            return -1  # division by 2 reduces entropy by 1 bit
        else:
            # Compute τ for odd step
            m = 3 * n + 1
            tau = 0
            while m % 2 == 0:
                tau += 1
                m //= 2
            return np.log2(3) - tau

    # Compute entropy changes
    numbers = range(1, max_n + 1)
    entropy_changes = [entropy_change(n) for n in numbers]

    # Create figure
    plt.figure(figsize=(12, 8))
    plt.style.use("dark_background")

    # Plot entropy changes
    plt.scatter(numbers, entropy_changes, c=entropy_changes, cmap="coolwarm", alpha=0.6)

    # Add zero line
    plt.axhline(y=0, color="white", linestyle="--", alpha=0.3)

    # Styling
    plt.title("Entropy Change per Collatz Step", fontsize=16, pad=20)
    plt.xlabel("Starting Number")
    plt.ylabel("Entropy Change (bits)")
    plt.colorbar(label="Entropy Change")
    plt.grid(True, alpha=0.2)

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.replace(".svg", ".pdf")
    plt.savefig(
        "figures/entropy_reduction.pdf", format="pdf", bbox_inches="tight", dpi=300
    )
    plt.close()


def create_compression_visualization(output_path="figures/compression_ratio.svg"):
    """Create a visualization of compression ratios."""
    # Create figure
    plt.figure(figsize=(12, 6))
    plt.style.use("dark_background")

    # Create boxes
    plt.fill_between([-0.4, 0.4], [-0.4, -0.4], [0.4, 0.4], color="#FF6B6B", alpha=0.7)
    plt.text(0, 0, "Input\nBits", ha="center", va="center", color="white", fontsize=12)

    plt.fill_between([1.6, 2.0], [-0.2, -0.2], [0.2, 0.2], color="#4ECDC4", alpha=0.7)
    plt.text(
        1.8, 0, "Output\nBits", ha="center", va="center", color="white", fontsize=12
    )

    # Add arrow
    plt.arrow(0.5, 0, 1.0, 0, head_width=0.1, head_length=0.1, fc="white", ec="white")

    # Add compression ratio label
    plt.text(
        1.0,
        0.5,
        "Compression\nRatio < 1",
        ha="center",
        va="bottom",
        color="white",
        fontsize=12,
    )

    # Styling
    plt.xlim(-0.6, 2.2)
    plt.ylim(-0.6, 0.6)
    plt.axis("off")
    plt.title("Information Compression in Collatz Steps", pad=20)

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.replace(".svg", ".pdf")
    plt.savefig(
        "figures/compression_ratio.pdf", format="pdf", bbox_inches="tight", dpi=300
    )
    plt.close()


def plot_bit_evolution(n=27, steps=100, output_path="figures/bit_evolution.svg"):
    """Create a visualization of bit pattern evolution."""

    def collatz_step(n):
        return n // 2 if n % 2 == 0 else 3 * n + 1

    # Generate sequence
    sequence = [n]
    for _ in range(steps):
        n = collatz_step(n)
        sequence.append(n)
        if n == 1:
            break

    # Convert to bit patterns
    bit_patterns = [format(n, "b") for n in sequence]
    max_length = max(len(p) for p in bit_patterns)
    bit_patterns = [p.zfill(max_length) for p in bit_patterns]

    # Create binary matrix
    matrix = np.array([[int(b) for b in p] for p in bit_patterns])

    # Create figure
    plt.figure(figsize=(15, 10))
    plt.style.use("dark_background")

    # Plot heatmap with custom colormap
    cmap = plt.cm.get_cmap("viridis")
    sns.heatmap(matrix, cmap=cmap, cbar=True, cbar_kws={"label": "Bit Value"})

    # Styling
    plt.title("Bit Pattern Evolution in Collatz Sequence", fontsize=16, pad=20)
    plt.xlabel("Bit Position")
    plt.ylabel("Step")

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.replace(".svg", ".pdf")
    plt.savefig("figures/bit_evolution.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Create all visualizations
    plot_entropy_reduction()
    create_compression_visualization()
    plot_bit_evolution()
