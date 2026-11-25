"""Build ATP tennis matrices and inspect the original directed match graph."""

import networkx as nx
import numpy as np
import pandas as pd


def load_tennis_data(data_dir):
    """Read the two course CSVs and keep players in node ID order."""
    nodes = pd.read_csv(data_dir / "nodes.csv").sort_values("node_id", ignore_index=True)
    edges = pd.read_csv(data_dir / "edges.csv")
    return nodes, edges


def build_adjacency(n, edges):
    """Set A[i, j] to 1 when player i lost to player j at least once."""
    sources = edges["source"].to_numpy()
    targets = edges["target"].to_numpy()
    adjacency = np.zeros((n, n), dtype=int)
    adjacency[sources, targets] = 1
    return adjacency


def build_transition(adjacency):
    """Normalize the nonempty rows and replace empty rows with uniform probabilities."""
    out_degree = adjacency.sum(axis=1)
    dangling = out_degree == 0
    transition = np.full(adjacency.shape, 1.0 / len(adjacency))
    transition[~dangling] = adjacency[~dangling] / out_degree[~dangling, None]
    return transition, out_degree, dangling


def inspect_components(graph):
    """Return SCCs by size and the components with no edge leaving them."""
    components = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    closed = [
        component for component in components
        if all(
            target in component
            for player in component
            for target in graph.successors(player)
        )
    ]
    return components, closed


def plot_outgoing_row(adjacency, player_names, source):
    """Draw all outgoing match arrows for the player used in the worked example."""
    import matplotlib.pyplot as plt

    targets = np.flatnonzero(adjacency[source])
    if not targets.size:
        raise ValueError("The worked example needs a player with outgoing match arrows.")

    graph = nx.DiGraph()
    graph.add_edges_from((source, target) for target in targets)
    positions = {source: (0, 0.5)}
    positions.update(zip(targets, [(1, height) for height in np.linspace(0, 1, len(targets))]))

    fig, ax = plt.subplots(figsize=(8.0, 3.1), layout="constrained")
    nx.draw_networkx_edges(
        graph, positions, ax=ax, arrows=True, arrowsize=20,
        node_size=2100, width=1.8, edge_color="#576675",
    )
    nx.draw_networkx_nodes(
        graph, positions, ax=ax, node_size=2100,
        node_color=["#2459A6" if node == source else "#237A58" for node in graph],
    )
    nx.draw_networkx_labels(
        graph, positions, ax=ax, labels={node: str(node) for node in graph},
        font_color="white", font_weight="bold",
    )
    for node, (x, y) in positions.items():
        ax.text(x, y - 0.19, player_names[node], ha="center", va="top")
    ax.set(title="Martin Damm’s outgoing match arrows", xlim=(-0.28, 1.30), ylim=(-0.35, 1.25))
    ax.set_axis_off()
    return fig
