"""
covariance_basics.py

Basic covariance matrix plots for the kappa-kinisi project.
Builds intuition for covariance, eigenvalues, and the condition number.

Run:  python covariance_basics.py
"""

import numpy as np
import matplotlib.pyplot as plt


# Use the same random numbers every time
rng = np.random.default_rng(42)


def make_data():
    """Make a cloud of 1000 points where x and y are correlated."""
    x = rng.normal(0, 1, 1000)
    y = rng.normal(0, 1, 1000)
    y = 0.8 * x + 0.6 * y  # mixing in x makes y correlated with x
    return np.column_stack([x, y])


def plot_scatter(data):
    """Plot the data cloud. Correlation shows up as a tilt."""
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.3, s=10)
    plt.title("Correlated 2D data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.savefig("covariance_scatter.png", dpi=150)
    plt.close()
    print("Saved covariance_scatter.png")


def plot_heatmap(data):
    """Plot the covariance matrix as a grid of numbers."""
    # np.cov needs the data transposed (variables as rows)
    cov = np.cov(data.T)

    plt.figure(figsize=(5, 4))
    plt.imshow(cov, cmap="coolwarm")
    plt.colorbar(label="covariance")

    # write the value inside each cell
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cov[i, j]:.2f}", ha="center", va="center")

    plt.xticks([0, 1], ["x", "y"])
    plt.yticks([0, 1], ["x", "y"])
    plt.title("Covariance matrix")
    plt.savefig("covariance_heatmap.png", dpi=150)
    plt.close()
    print("Saved covariance_heatmap.png")
    print("  diagonal = variances, off-diagonal = covariance")


def plot_condition_number():
    """Show how the condition number grows as the data is squashed."""
    x = rng.normal(0, 1, 1000)
    y = rng.normal(0, 1, 1000)

    # squash y by smaller and smaller amounts
    squash = [1.0, 0.5, 0.1, 0.05, 0.01, 0.001]
    kappa = []

    for s in squash:
        data = np.column_stack([x, s * y])
        cov = np.cov(data.T)
        kappa.append(np.linalg.cond(cov))  # condition number

    plt.figure(figsize=(6, 5))
    plt.loglog(squash, kappa, "o-")
    plt.xlabel("squash factor (smaller = thinner cloud)")
    plt.ylabel("condition number")
    plt.title("Thinner data means a bigger condition number")
    plt.grid(alpha=0.3, which="both")
    plt.savefig("condition_number.png", dpi=150)
    plt.close()
    print("Saved condition_number.png")
    for s, k in zip(squash, kappa):
        print(f"  squash {s} -> condition number {k:.1e}")


# run everything
data = make_data()
plot_scatter(data)
plot_heatmap(data)
plot_condition_number()
print("Done.")
