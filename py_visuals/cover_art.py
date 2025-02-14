import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import seaborn as sns
from pathlib import Path


def create_cover_art(output_path="figures/cover_art.svg"):
    """Create an epic cover visualization combining key aspects of the proof."""
    # Create figure with dark theme
    fig = plt.figure(figsize=(24, 18))
    plt.style.use("dark_background")

    # Enhanced color palette inspired by modern cybersecurity visuals
    colors = {
        "bg": "#0D1117",  # Deep space black
        "primary": "#58A6FF",  # Electric blue
        "secondary": "#FF6B6B",  # Neon coral
        "accent": "#4ECDC4",  # Cyber mint
        "highlight": "#FFD93D",  # Digital gold
        "matrix": "#00FF41",  # Matrix green
        "purple": "#BD93F9",  # Cyber purple
        "warning": "#FF5555",  # Alert red
    }

    # Set background color
    fig.patch.set_facecolor(colors["bg"])

    # Create a 2x2 grid with spacing
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # 1. Top left: Enhanced Spiral Convergence with 3D effect
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.set_facecolor(colors["bg"])

    # Generate 3D spiral data
    theta = np.linspace(0, 12 * np.pi, 2000)
    r = np.exp(-theta / 15)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.exp(-theta / 8)  # Height component

    # Create color gradient
    colors_spiral = plt.cm.viridis(np.linspace(0, 1, len(theta)))

    # Plot 3D spiral with glow effect
    for alpha in [0.1, 0.2, 0.3]:
        ax1.plot(x, y, z, color=colors["primary"], alpha=alpha, linewidth=3)
    ax1.plot(x, y, z, color=colors["primary"], alpha=0.8, linewidth=2)

    # Add key points with glowing spheres
    key_points = [(0, 0, 0), (0.2, 0, 0.1), (0.1, 0.1, 0.05)]
    labels = ["1", "2", "4"]
    for point, label in zip(key_points, labels):
        # Glow effect
        for size in [100, 80, 60, 40]:
            ax1.scatter(*point, s=size, color=colors["highlight"], alpha=0.2)
        ax1.scatter(*point, s=30, color=colors["highlight"], alpha=1)
        ax1.text(point[0], point[1], point[2], label, color=colors["highlight"])

    # Add mathematical annotations
    ax1.text2D(
        0.05,
        0.95,
        "$T_{odd}(n) = \\frac{3n + 1}{2^{\\tau(n)}}$",
        transform=ax1.transAxes,
        fontsize=12,
        color=colors["accent"],
    )

    ax1.set_title(
        "Global Convergence & Cycle Structure", pad=20, fontsize=16, color="white"
    )

    # Remove axes for cleaner look
    ax1.set_axis_off()

    # 2. Top right: Enhanced Bit Pattern Cascade
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(colors["bg"])

    # Generate sophisticated bit pattern data
    n_steps, n_bits = 30, 40
    patterns = np.zeros((n_steps, n_bits))

    # Create interesting initial pattern
    patterns[0, n_bits // 2 - 5 : n_bits // 2 + 5] = 1

    # Generate avalanche effect
    for i in range(1, n_steps):
        prev_pattern = patterns[i - 1]
        new_pattern = np.zeros(n_bits)
        for j in range(n_bits):
            if prev_pattern[j] == 1:
                # Create sophisticated spreading pattern
                spread = np.exp(-np.abs(np.arange(n_bits) - j) / 3)
                new_pattern += spread * (np.random.random(n_bits) < 0.7)
        patterns[i] = (new_pattern > 0.3).astype(float)

    # Create custom colormap with cyber theme
    custom_cmap = plt.cm.RdYlBu_r

    # Plot enhanced heatmap with glow effect
    sns.heatmap(patterns, cmap=custom_cmap, cbar=False, ax=ax2)

    # Add information theory equation
    ax2.text(
        0.05,
        0.95,
        "$\\Delta H = \\log_2(3) - \\tau(n) + \\epsilon(n)$",
        transform=ax2.transAxes,
        fontsize=12,
        color=colors["accent"],
    )

    ax2.set_title("Bit Pattern Avalanche & Information Flow", pad=20, fontsize=16)

    # 3. Bottom left: Enhanced Phase Space
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(colors["bg"])

    # Generate enhanced phase space data
    t = np.linspace(0, 30, 3000)
    x = np.cos(t) * np.exp(-t / 20) * (1 + 0.2 * np.sin(3 * t))
    y = np.sin(t) * np.exp(-t / 20) * (1 + 0.2 * np.cos(3 * t))

    # Create gradient effect
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="viridis", alpha=0.8, linewidth=2)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax3.add_collection(lc)

    # Add measure theory equation
    ax3.text(
        0.05,
        0.95,
        "$d(T^{-1}(A)) = d(A)$",
        transform=ax3.transAxes,
        fontsize=12,
        color=colors["accent"],
    )

    # Add glowing boundary
    theta = np.linspace(0, 2 * np.pi, 200)
    for width in [3, 2, 1]:
        ax3.plot(
            np.cos(theta),
            np.sin(theta),
            "--",
            color=colors["purple"],
            alpha=0.3,
            linewidth=width,
        )

    ax3.set_title("Ergodic Behavior & Measure Preservation", pad=20, fontsize=16)
    ax3.set_aspect("equal")
    ax3.set_axis_off()

    # 4. Bottom right: Entropy Cascade
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(colors["bg"])

    # Generate entropy cascade data
    x = np.linspace(0, 15, 500)
    base = 8 * np.exp(-x / 5)
    oscillation = 0.8 * np.sin(2 * x) * np.exp(-x / 6)
    noise = np.random.normal(0, 0.15, len(x))
    entropy = base + oscillation + noise

    # Create gradient fill with cyber theme
    gradient = np.linspace(0, 1, len(x))
    for i in range(len(x) - 1):
        ax4.fill_between(
            x[i : i + 2],
            entropy[i : i + 2],
            alpha=0.3,
            color=plt.cm.viridis(gradient[i]),
        )

    # Add multiple layers for glow effect
    alphas = [0.1, 0.2, 0.3, 0.8]
    widths = [4, 3, 2, 1.5]
    for alpha, width in zip(alphas, widths):
        ax4.plot(x, entropy, color=colors["accent"], alpha=alpha, linewidth=width)

    # Add probability equation
    ax4.text(
        0.05,
        0.95,
        "$P(\\tau = k) = 2^{-k} + O(n^{-1/2})$",
        transform=ax4.transAxes,
        fontsize=12,
        color=colors["accent"],
    )

    ax4.set_title("Entropy Reduction & Compression", pad=20, fontsize=16)
    ax4.set_axis_off()

    # Main title with enhanced styling
    plt.suptitle(
        "The Collatz Conjecture:\nA Cryptographic Perspective",
        fontsize=32,
        y=0.95,
        color="white",
        fontweight="bold",
        family="monospace",
    )

    # Add subtitle
    plt.figtext(
        0.5,
        0.91,
        "Bridging Number Theory and Modern Cryptography",
        ha="center",
        color=colors["accent"],
        fontsize=18,
        family="monospace",
    )

    # Save with high quality
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.replace(".svg", ".pdf")
    plt.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        facecolor=colors["bg"],
        edgecolor="none",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    create_cover_art()
