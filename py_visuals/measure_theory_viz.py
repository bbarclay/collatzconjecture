import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
from pathlib import Path
import os


def plot_tau_distribution(max_n=1000, output_path="figures/tau_distribution.svg"):
    """Create a visualization of τ distribution."""

    def compute_tau(n):
        if n % 2 == 0:
            return 0
        m = 3 * n + 1
        tau = 0
        while m % 2 == 0:
            tau += 1
            m //= 2
        return tau

    # Compute τ values
    odd_numbers = range(1, max_n, 2)
    tau_values = [compute_tau(n) for n in odd_numbers]

    # Create figure
    plt.figure(figsize=(12, 8))
    plt.style.use("dark_background")

    # Plot histogram
    sns.histplot(tau_values, discrete=True, color="#4ECDC4")

    # Plot theoretical distribution
    max_tau = max(tau_values)
    theoretical_x = np.arange(1, max_tau + 1)
    theoretical_y = [max_n / 4 * (1 / 2) ** (k - 1) for k in theoretical_x]
    plt.plot(theoretical_x, theoretical_y, "r--", label="Theoretical Distribution")

    # Styling
    plt.title("Distribution of τ Values", fontsize=16, pad=20)
    plt.xlabel("τ Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.2)
    plt.legend()

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.replace(".svg", ".pdf")
    plt.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        facecolor="black",
        edgecolor="none",
    )
    plt.close()


def create_ergodic_visualization(output_path="figures/ergodic_property.svg"):
    """Create a visualization of the ergodic property."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.style.use("dark_background")

    # Create circle (phase space)
    circle = plt.Circle((0, 0), 1, fill=False, color="white", alpha=0.3)
    ax.add_artist(circle)

    # Create trajectory (spiral)
    t = np.linspace(0, 20 * np.pi, 1000)
    r = np.exp(-t / 20)
    x = r * np.cos(t)
    y = r * np.sin(t)

    # Plot trajectory with color gradient
    points = np.array([x, y]).T
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    lc = LineCollection(segments, cmap="viridis", alpha=0.6)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)

    # Add random points for residue classes
    np.random.seed(42)
    for i in range(50):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, 1)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        plt.plot(x, y, "o", color=plt.cm.viridis(i / 50), markersize=3)

    # Styling
    plt.title("Ergodic Trajectory in Phase Space", fontsize=16, pad=20)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    plt.grid(True, alpha=0.2)

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.replace(".svg", ".pdf")
    plt.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        facecolor="black",
        edgecolor="none",
    )
    plt.close()


def plot_vertical_structure(max_n=100, output_path="figures/vertical_structure.svg"):
    """Create a visualization of the vertical structure of trajectories."""

    def collatz_sequence(n, max_steps=1000):
        sequence = [n]
        while n != 1 and len(sequence) < max_steps:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            sequence.append(n)
        return sequence

    # Generate trajectories for carefully chosen starting points
    start_points = list(range(1, max_n + 1, 2))  # Odd numbers
    trajectories = [collatz_sequence(n) for n in start_points]

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.style.use("dark_background")

    # Plot trajectories with color gradient
    for i, traj in enumerate(trajectories):
        x = np.arange(len(traj))
        y = np.log2(traj)
        points = np.array([x, y]).T
        segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
        lc = LineCollection(segments, cmap="viridis", alpha=0.3)
        lc.set_array(np.linspace(0, 1, len(segments)))
        ax.add_collection(lc)

    # Add reference lines
    x_max = max(len(t) for t in trajectories)
    plt.axhline(
        y=np.log2(1), color="red", linestyle="--", alpha=0.3, label="Terminal Value (1)"
    )
    plt.axhline(
        y=np.log2(4), color="yellow", linestyle="--", alpha=0.3, label="Cycle Entry (4)"
    )

    # Styling
    plt.title("Vertical Structure of Collatz Trajectories", fontsize=16, pad=20)
    plt.xlabel("Step")
    plt.ylabel("log₂(Value)")
    plt.grid(True, alpha=0.2)
    plt.legend()

    # Set axis limits
    plt.xlim(0, x_max)
    y_max = max(max(np.log2(t)) for t in trajectories)
    plt.ylim(0, y_max * 1.1)

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.replace(".svg", ".pdf")
    plt.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        facecolor="black",
        edgecolor="none",
    )
    plt.close()


if __name__ == "__main__":
    # Create all visualizations
    plot_tau_distribution()
    create_ergodic_visualization()
    plot_vertical_structure()
