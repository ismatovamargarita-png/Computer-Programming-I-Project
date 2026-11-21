"""Task 3: run the original ATP tennis random walk from two players."""

import argparse
import csv
from pathlib import Path
from typing import NamedTuple

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
STARTING_PLAYERS = ("Novak Djokovic", "Rohan Bopanna")
PLOT_PLAYERS = (*STARTING_PLAYERS, "Alexander Zverev", "Martin Damm")
TOLERANCE = 1e-10
MAX_ITERATIONS = 5000


class WalkResult(NamedTuple):
    final: np.ndarray
    iterations: int
    history: np.ndarray
    changes: np.ndarray
    converged: bool


def load_network(data_dir=None):
    """Read Task 1's saved matrices and map player names to their matrix indices."""
    if data_dir is None:
        data_dir = PROJECT_DIR / "data"
    with np.load(data_dir / "matrices.npz") as matrices:
        adjacency = matrices["A"]
        transition = matrices["P"]

    with (data_dir / "nodes.csv").open(encoding="utf-8", newline="") as file:
        player_ids = {
            row["label"]: int(row["node_id"]) for row in csv.DictReader(file)
        }

    n = len(player_ids)
    if adjacency.shape != (n, n) or transition.shape != (n, n):
        raise ValueError("The saved matrices do not match nodes.csv.")
    if set(player_ids.values()) != set(range(n)):
        raise ValueError("The player IDs must match the matrix indices.")
    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1):
        raise ValueError("P must have nonnegative entries and row sums of one.")
    return adjacency, transition, player_ids


def power_iteration(transition, initial, tol=TOLERANCE, max_iter=MAX_ITERATIONS):
    """Apply the matrix one step at a time. History includes the initial state."""
    current = np.array(initial, dtype=float, copy=True)
    history = [current]
    changes = []

    for _ in range(max_iter):
        updated = current @ transition
        change = np.abs(updated - current).sum()
        history.append(updated)
        changes.append(change)
        current = updated
        if change < tol:
            break

    return WalkResult(
        final=current,
        iterations=len(changes),
        history=np.array(history),
        changes=np.array(changes),
        converged=bool(changes and changes[-1] < tol),
    )


def run_from_starting_players(transition, player_ids):
    """Use a point mass at each starting player and the same stopping rule."""
    runs = {}
    for name in STARTING_PLAYERS:
        initial = np.zeros(len(player_ids))
        initial[player_ids[name]] = 1.0
        runs[name] = power_iteration(transition, initial)
    return runs


def save_figures(runs, player_ids, figures_dir):
    """Save the player trajectories and the per-step convergence comparison."""
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    player_colors = dict(zip(PLOT_PLAYERS, ("#2459A6", "#863E8D", "#BA500A", "#237A58")))
    max_steps = max(result.iterations for result in runs.values())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout="constrained")

    for ax, (name, result) in zip(axes, runs.items()):
        for player, color in player_colors.items():
            values = result.history[:, player_ids[player]]
            # Zero probability has no position on a logarithmic axis.
            ax.semilogy(np.where(values > 0, values, np.nan), color=color, label=player)
        ax.scatter(0, 1, color=player_colors[name], s=22, zorder=3)
        ax.set(title=f"Start at {name}", xlabel="Step", xlim=(-0.5, max_steps))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Player probability (log scale)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Original walk from two starting players")
    fig.savefig(figures_dir / "task3_player_probabilities.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for name, result in runs.items():
        ax.semilogy(
            range(1, result.iterations + 1),
            result.changes,
            color=player_colors[name],
            label=name,
        )
    ax.axhline(TOLERANCE, color="gray", linestyle="--", label="Stopping threshold")
    ax.set(
        title="How quickly the original walk settles",
        xlabel="Update number",
        ylabel="L1 change between successive distributions",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(figures_dir / "task3_convergence.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir", type=Path, default=PROJECT_DIR / "figures",
        help="Directory for the two Task 3 figures.",
    )
    args = parser.parse_args()

    _, transition, player_ids = load_network()
    runs = run_from_starting_players(transition, player_ids)
    for name, result in runs.items():
        print(f"{name}: {result.iterations} steps, converged = {result.converged}")
        print(f"  Last L1 change: {result.changes[-1]:.3e}")
    first, second = runs.values()
    print(f"L1 difference between final distributions: {np.abs(first.final - second.final).sum():.3e}")

    save_figures(runs, player_ids, args.figures_dir)
    print(f"Saved the two Task 3 figures in {args.figures_dir}")


if __name__ == "__main__":
    main()
