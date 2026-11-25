"""Task 3: run the original ATP tennis random walk from two players."""

import argparse
import csv
from pathlib import Path
from typing import NamedTuple

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
STARTING_PLAYERS = ("Novak Djokovic", "Rohan Bopanna")
PLOT_PLAYERS = (*STARTING_PLAYERS, "Alexander Zverev", "Martin Damm")
PLAYER_COLORS = dict(zip(PLOT_PLAYERS, ("#2459A6", "#863E8D", "#BA500A", "#237A58")))
PLAYER_LINE_STYLES = dict(zip(PLOT_PLAYERS, ("-", "--", "-.", ":")))
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
    """Return the last vector, update count, histories, and stopping status.

    history[k] is x_k, with history[0] holding the initial distribution.
    changes[k - 1] is the L1 distance between x_k and x_(k - 1).
    """
    if not np.isfinite(tol) or tol <= 0 or max_iter < 1:
        raise ValueError("The tolerance and iteration cap must be positive.")
    current = np.array(initial, dtype=float, copy=True)
    if (
        current.shape != (transition.shape[0],)
        or np.any(current < 0)
        or not np.isclose(current.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("The initial vector must be a probability distribution.")
    history = [current]
    changes = []

    for _ in range(max_iter):
        # Right-multiplication redistributes every player's current probability.
        updated = current @ transition
        # Absolute values prevent gains and losses at different players cancelling.
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
        converged=bool(changes[-1] < tol),
    )


def run_from_starting_players(transition, player_ids):
    """Use a point mass at each starting player and the same stopping rule."""
    runs = {}
    for name in STARTING_PLAYERS:
        initial = np.zeros(len(player_ids))
        initial[player_ids[name]] = 1.0
        result = power_iteration(transition, initial)
        # Every saved row should still represent one unit of probability.
        np.testing.assert_allclose(result.history.sum(axis=1), 1, atol=1e-12, rtol=0)
        if np.any(result.history < 0):
            raise ValueError("The walk produced negative probabilities.")
        runs[name] = result
    return runs


def plot_player_probabilities(runs, player_ids):
    """Return the figure of selected player probabilities for both starts."""
    import matplotlib.pyplot as plt

    max_steps = max(result.iterations for result in runs.values())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout="constrained")
    linear_threshold = 1e-7
    tick_values = (0, 1e-6, 1e-4, 1e-2, 1)
    tick_labels = ("0", "10⁻⁶", "10⁻⁴", "10⁻²", "1")

    for ax, (name, result) in zip(axes, runs.items()):
        ax.set_yscale("symlog", linthresh=linear_threshold)
        for player, color in PLAYER_COLORS.items():
            values = result.history[:, player_ids[player]]
            # A linear band around zero lets us plot the original values, including zeros.
            ax.plot(
                range(len(values)), values, color=color, label=player,
                linestyle=PLAYER_LINE_STYLES[player], linewidth=1.8,
            )
        ax.scatter(0, 1, color=PLAYER_COLORS[name], s=25, zorder=3)
        ax.axhline(0, color="#64748B", linewidth=0.8, zorder=0)
        ax.set(title=f"Start at {name}", xlabel="Step", xlim=(-0.5, max_steps + 0.5))
        ax.set_yticks(tick_values, tick_labels)
        ax.grid(alpha=0.2)
    axes[0].set_ylim(-0.2 * linear_threshold, 1.25)
    axes[0].set_ylabel("Player probability (log above 10⁻⁷)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Original walk from two starting players")
    return fig


def plot_convergence(runs, tolerance=TOLERANCE):
    """Return the figure of the per-update change and stopping threshold."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for name, result in runs.items():
        ax.semilogy(
            range(1, result.iterations + 1),
            result.changes,
            color=PLAYER_COLORS[name],
            linestyle=PLAYER_LINE_STYLES[name],
            linewidth=1.9,
            label=f"Start at {name}",
        )
        if result.converged and result.changes[-1] > 0:
            endpoint = (result.iterations, result.changes[-1])
            ax.scatter(*endpoint, color=PLAYER_COLORS[name], s=26, zorder=4)
            ax.annotate(
                f"{result.iterations} steps", endpoint, xytext=(-6, -14),
                textcoords="offset points", ha="right", va="top",
                fontsize=9, color=PLAYER_COLORS[name],
            )
    ax.axhline(tolerance, color="gray", linestyle=":", label=f"Tolerance {tolerance:.0e}")
    ax.set(
        title="How quickly the original walk settles",
        xlabel="Update number",
        ylabel="Total absolute change (L1, log scale)",
    )
    ax.grid(alpha=0.2)
    ax.margins(y=0.1)
    ax.set_ylim(top=3)
    ax.legend()
    return fig


def print_walk_summary(runs):
    """Report each run's starting behavior, stopping status, and final gap."""
    for name, result in runs.items():
        status = "threshold reached" if result.converged else "iteration cap reached"
        print(f"{name}: {result.iterations} steps ({status})")
        print(f"  Players with positive probability after step 1: {np.count_nonzero(result.history[1])}")
        print(f"  Last L1 change: {result.changes[-1]:.3e}")
    first, second = runs.values()
    print(f"L1 difference between final distributions: {np.abs(first.final - second.final).sum():.3e}")


def save_figures(runs, player_ids, figures_dir):
    """Save both Task 3 figures for the report and close them."""
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    trajectory = plot_player_probabilities(runs, player_ids)
    trajectory.savefig(figures_dir / "task3_player_probabilities.png", dpi=200)
    plt.close(trajectory)
    convergence = plot_convergence(runs)
    convergence.savefig(figures_dir / "task3_convergence.png", dpi=200)
    plt.close(convergence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir", type=Path, default=PROJECT_DIR / "figures",
        help="Directory for the two Task 3 figures.",
    )
    args = parser.parse_args()
    _, transition, player_ids = load_network()
    runs = run_from_starting_players(transition, player_ids)
    print_walk_summary(runs)

    save_figures(runs, player_ids, args.figures_dir)
    print(f"Saved the two Task 3 figures in {args.figures_dir}")


if __name__ == "__main__":
    main()
