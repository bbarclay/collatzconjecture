import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_transformation_diagram(output_path="figures/transformation_phases.svg"):
    """Create a stunning visual diagram of the three-phase Collatz transformation."""
    # Create figure
    plt.figure(figsize=(12, 6))
    plt.style.use("dark_background")

    # Define colors
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    # Create boxes
    phases = ["Expansion (×3)", "Mixing (+1)", "Compression (÷2τ)"]
    for i, (phase, color) in enumerate(zip(phases, colors)):
        plt.fill_between(
            [i - 0.4, i + 0.4], [-0.4] * 2, [0.4] * 2, color=color, alpha=0.7
        )
        plt.text(i, 0, phase, ha="center", va="center", fontsize=12, color="white")

    # Add arrows
    for i in range(len(phases) - 1):
        plt.arrow(
            i + 0.45,
            0,
            0.1,
            0,
            head_width=0.1,
            head_length=0.05,
            fc="white",
            ec="white",
        )

    # Styling
    plt.xlim(-0.8, len(phases) - 0.2)
    plt.ylim(-0.6, 0.6)
    plt.axis("off")
    plt.title("Three-Phase Collatz Transformation", pad=20)

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
        facecolor="black",
        edgecolor="none",
    )
    plt.close()


def plot_trajectory_tree(
    start_n=27, depth=5, output_path="figures/trajectory_tree.svg"
):
    """Create a visual tree diagram of a Collatz trajectory."""

    def collatz_step(n):
        return n // 2 if n % 2 == 0 else 3 * n + 1

    # Generate trajectory
    trajectory = [start_n]
    current = start_n
    while len(trajectory) < depth and current != 1:
        current = collatz_step(current)
        trajectory.append(current)

    # Create figure
    plt.figure(figsize=(15, 10))
    plt.style.use("dark_background")

    # Plot trajectory
    x = np.arange(len(trajectory))
    y = np.array(trajectory)
    plt.plot(x, y, "w-", linewidth=2, alpha=0.7)
    plt.scatter(x, y, c=np.log(y), cmap="viridis", s=200, zorder=5)

    # Add value labels
    for i, txt in enumerate(trajectory):
        plt.annotate(
            str(txt),
            (x[i], y[i]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color="white",
        )

    # Styling
    plt.title(f"Collatz Trajectory Tree Starting at n={start_n}", fontsize=16, pad=20)
    plt.grid(True, alpha=0.2)
    plt.xlabel("Step")
    plt.ylabel("Value")

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
        facecolor="black",
        edgecolor="none",
    )
    plt.close()


def create_bit_pattern_visualization(n_bits=8, output_path="figures/bit_patterns.svg"):
    """Create a visualization of bit pattern evolution in Collatz steps."""
    # Create sample bit patterns showing interesting transformations
    patterns = [
        ("11111111", "11111111", "00000001"),  # All ones case
        ("10101010", "11111111", "00000011"),  # Alternating pattern
        ("11001100", "10101010", "00000111"),  # Paired bits
        ("11110000", "11001100", "00001111"),  # Half and half
        ("10000000", "11110000", "00011111"),  # Single bit cascade
    ]

    # Create figure
    plt.figure(figsize=(15, 8))
    plt.style.use("dark_background")

    # Define vibrant colors
    colors = {
        "1": "#FF6B6B",  # Bright red for 1s
        "0": "#2C3E50",  # Dark blue for 0s
        "bg": "#1a1a1a",  # Darker background
    }

    # Plot patterns
    for i, (original, intermediate, final) in enumerate(patterns):
        y_pos = len(patterns) - i - 1

        # Plot original bits with glow effect
        for j, bit in enumerate(original):
            # Add glow effect
            if bit == "1":
                plt.fill_between(
                    [j - 0.45 + 0, j + 0.45 + 0],
                    [y_pos - 0.45] * 2,
                    [y_pos + 0.45] * 2,
                    color=colors["1"],
                    alpha=0.3,
                )
            plt.fill_between(
                [j - 0.4 + 0, j + 0.4 + 0],
                [y_pos - 0.4] * 2,
                [y_pos + 0.4] * 2,
                color=colors["1"] if bit == "1" else colors["0"],
                alpha=0.7,
            )
            plt.text(
                j + 0,
                y_pos,
                bit,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

        # Add arrows showing transformation
        plt.arrow(
            8.5,
            y_pos,
            1,
            0,
            head_width=0.1,
            head_length=0.2,
            fc="white",
            ec="white",
            alpha=0.5,
        )
        plt.arrow(
            18.5,
            y_pos,
            1,
            0,
            head_width=0.1,
            head_length=0.2,
            fc="white",
            ec="white",
            alpha=0.5,
        )

        # Plot intermediate bits
        for j, bit in enumerate(intermediate):
            if bit == "1":
                plt.fill_between(
                    [j - 0.45 + 10, j + 0.45 + 10],
                    [y_pos - 0.45] * 2,
                    [y_pos + 0.45] * 2,
                    color=colors["1"],
                    alpha=0.3,
                )
            plt.fill_between(
                [j - 0.4 + 10, j + 0.4 + 10],
                [y_pos - 0.4] * 2,
                [y_pos + 0.4] * 2,
                color=colors["1"] if bit == "1" else colors["0"],
                alpha=0.7,
            )
            plt.text(
                j + 10,
                y_pos,
                bit,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

        # Plot final bits
        for j, bit in enumerate(final):
            if bit == "1":
                plt.fill_between(
                    [j - 0.45 + 20, j + 0.45 + 20],
                    [y_pos - 0.45] * 2,
                    [y_pos + 0.45] * 2,
                    color=colors["1"],
                    alpha=0.3,
                )
            plt.fill_between(
                [j - 0.4 + 20, j + 0.4 + 20],
                [y_pos - 0.4] * 2,
                [y_pos + 0.4] * 2,
                color=colors["1"] if bit == "1" else colors["0"],
                alpha=0.7,
            )
            plt.text(
                j + 20,
                y_pos,
                bit,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    # Add labels with better styling
    plt.text(
        4,
        len(patterns),
        "Original Pattern",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    plt.text(
        14,
        len(patterns),
        "After ×3 + 1",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    plt.text(
        24,
        len(patterns),
        "After ÷2τ",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold",
    )

    # Add transformation labels
    plt.text(
        9.5, -0.5, "Expansion", ha="center", va="center", color="#4ECDC4", fontsize=10
    )
    plt.text(
        19.5,
        -0.5,
        "Compression",
        ha="center",
        va="center",
        color="#FF6B6B",
        fontsize=10,
    )

    # Styling
    plt.xlim(-2, 28)
    plt.ylim(-1, len(patterns) + 1)
    plt.axis("off")
    plt.title(
        "Bit Pattern Evolution in Collatz Steps", pad=20, fontsize=14, fontweight="bold"
    )

    # Set figure background color
    plt.gca().set_facecolor(colors["bg"])
    plt.gcf().set_facecolor(colors["bg"])

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
        facecolor=colors["bg"],
        edgecolor="none",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    # Create all visualizations
    create_transformation_diagram()
    plot_trajectory_tree()
    create_bit_pattern_visualization()
