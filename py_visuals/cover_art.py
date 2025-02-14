import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import seaborn as sns
from pathlib import Path


def create_cover_art(output_path="figures/cover_art.svg"):
    """Create an epic cover visualization combining key aspects of the proof."""
    # Create figure with dark theme
    fig = plt.figure(figsize=(20, 15))
    plt.style.use("dark_background")

    # Custom colors
    colors = {
        "bg": "#0D1117",  # GitHub dark theme
        "primary": "#58A6FF",  # Bright blue
        "secondary": "#FF6B6B",  # Coral red
        "accent": "#4ECDC4",  # Turquoise
        "highlight": "#FFD93D",  # Gold
    }

    # Set background color
    fig.patch.set_facecolor(colors["bg"])

    # Create a 2x2 grid with spacing
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Top left: Trajectory spiral showing convergence
    ax1 = fig.add_subplot(gs[0, 0], projection="polar")
    ax1.set_facecolor(colors["bg"])

    # Generate enhanced spiral data
    theta = np.linspace(0, 15 * np.pi, 2000)
    r = np.exp(-theta / 12) * (1 + 0.3 * np.sin(5 * theta))
    colors_spiral = plt.cm.viridis(np.linspace(0, 1, len(theta)))

    # Plot spiral with enhanced color gradient and glow effect
    points = np.array([theta, r]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=colors_spiral, alpha=0.8, linewidth=2)
    ax1.add_collection(lc)

    # Add glowing convergence point
    for size in [150, 100, 50]:
        ax1.scatter(0, 0, c=colors["secondary"], s=size, alpha=0.3, zorder=5)
    ax1.scatter(0, 0, c=colors["secondary"], s=30, zorder=6, label="Convergence (1)")

    # Add cycle points with connecting arrows
    cycle_theta = [0, np.pi / 6, np.pi / 3]
    cycle_r = [0.2, 0.15, 0.1]
    ax1.scatter(
        cycle_theta, cycle_r, c=colors["highlight"], s=50, zorder=5, label="4→2→1 Cycle"
    )

    for i in range(len(cycle_theta) - 1):
        ax1.arrow(
            cycle_theta[i],
            cycle_r[i],
            (cycle_theta[i + 1] - cycle_theta[i]) / 2,
            (cycle_r[i + 1] - cycle_r[i]) / 2,
            head_width=0.05,
            head_length=0.05,
            fc=colors["highlight"],
            ec=colors["highlight"],
            alpha=0.7,
        )

    ax1.set_title("Global Convergence", pad=20, fontsize=16, color="white")
    ax1.legend(loc="upper right", framealpha=0.3)

    # 2. Top right: Enhanced bit pattern avalanche
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(colors["bg"])

    n_steps, n_bits = 25, 32
    patterns = np.zeros((n_steps, n_bits))
    patterns[0, n_bits // 2] = 1  # Initial single bit

    # Generate more realistic avalanche
    for i in range(1, n_steps):
        prev_pattern = patterns[i - 1]
        new_pattern = np.zeros(n_bits)
        for j in range(n_bits):
            if prev_pattern[j] == 1:
                # Affect neighboring bits with decreasing probability
                for k in range(max(0, j - 3), min(n_bits, j + 4)):
                    if np.random.random() < 0.7 / (1 + abs(k - j)):
                        new_pattern[k] = 1
        patterns[i] = new_pattern

    # Plot enhanced heatmap
    sns.heatmap(patterns, cmap="magma", cbar=False, ax=ax2)
    ax2.set_title("Avalanche Effect", pad=20, fontsize=16)
    ax2.set_xlabel("Bit Position", fontsize=12)
    ax2.set_ylabel("Step", fontsize=12)

    # 3. Bottom left: Enhanced entropy reduction
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(colors["bg"])

    # Generate more interesting entropy data
    x = np.linspace(0, 10, 300)
    base = 5 * np.exp(-x / 3)
    oscillation = 0.5 * np.sin(2 * x) * np.exp(-x / 4)
    noise = np.random.normal(0, 0.15, len(x))
    entropy = base + oscillation + noise

    # Create gradient fill
    gradient = np.linspace(0, 1, len(x))
    for i in range(len(x) - 1):
        ax3.fill_between(
            x[i : i + 2],
            entropy[i : i + 2],
            alpha=0.3,
            color=plt.cm.viridis(gradient[i]),
        )

    # Plot main line with glow effect
    for alpha in [0.1, 0.2, 0.3]:
        ax3.plot(x, entropy, color=colors["accent"], alpha=alpha, linewidth=4)
    ax3.plot(x, entropy, color=colors["accent"], alpha=0.8, linewidth=2)

    ax3.set_title("Entropy Reduction", pad=20, fontsize=16)
    ax3.set_xlabel("Time", fontsize=12)
    ax3.set_ylabel("Entropy", fontsize=12)
    ax3.grid(True, alpha=0.2)

    # 4. Bottom right: Enhanced phase space
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(colors["bg"])

    # Generate enhanced trajectory data
    t = np.linspace(0, 25, 2000)
    x = np.cos(t) * np.exp(-t / 15) * (1 + 0.2 * np.sin(3 * t))
    y = np.sin(t) * np.exp(-t / 15) * (1 + 0.2 * np.cos(3 * t))

    # Plot trajectory with gradient
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="viridis", alpha=0.8, linewidth=2)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax4.add_collection(lc)

    # Add enhanced phase space boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax4.plot(circle_x, circle_y, "--", color="white", alpha=0.3, linewidth=1)

    # Add glowing convergence point
    for size in [150, 100, 50]:
        ax4.scatter(0, 0, c=colors["secondary"], s=size, alpha=0.3, zorder=5)
    ax4.scatter(0, 0, c=colors["secondary"], s=30, zorder=6, label="Convergence")

    # Add enhanced cycle visualization
    cycle_x = [0.1, 0.05, 0.025]
    cycle_y = [0, 0.05, 0.025]
    ax4.scatter(cycle_x, cycle_y, c=colors["highlight"], s=50, label="4→2→1 Cycle")

    # Add glowing arrows between cycle points
    for i in range(len(cycle_x) - 1):
        arrow = FancyArrowPatch(
            (cycle_x[i], cycle_y[i]),
            (cycle_x[i + 1], cycle_y[i + 1]),
            arrowstyle="->",
            mutation_scale=15,
            color=colors["highlight"],
            alpha=0.7,
        )
        ax4.add_patch(arrow)

    ax4.set_title("Phase Space (No Larger Cycles)", pad=20, fontsize=16)
    ax4.set_xlabel("Re(z)", fontsize=12)
    ax4.set_ylabel("Im(z)", fontsize=12)
    ax4.grid(True, alpha=0.2)
    ax4.legend(loc="upper right", framealpha=0.3)
    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(-1.2, 1.2)
    ax4.set_aspect("equal")

    # Overall title with enhanced styling
    plt.suptitle(
        "The Collatz Conjecture:\nA Cryptographic Perspective",
        fontsize=28,
        y=0.95,
        color="white",
        fontweight="bold",
    )

    # Save figure with high quality
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
    create_cover_art()
