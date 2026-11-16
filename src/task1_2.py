
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# Locate the dataset
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

# Load the ATP tennis data
nodes = pd.read_csv(data_dir / "nodes.csv")
edges = pd.read_csv(data_dir / "edges.csv")

# Basic dataset information
print("ATP TENNIS NETWORK")
print("------------------")
print(f"Number of players: {len(nodes)}")
print(f"Number of connections: {len(edges)}")

print("\nFirst five players:")
print(nodes.head())

print("\nFirst five connections:")
print(edges.head())


# TASK 1: ADJACENCY MATRIX

n = len(nodes)

# Create a 1191 x 1191 matrix of zeros
A = np.zeros((n, n), dtype=int)

# Add directed connections
for source, target in edges[["source", "target"]].itertuples(index=False, name=None):
    A[source, target] = 1

# Verify the adjacency matrix
print("\nTASK 1: ADJACENCY MATRIX")
print("------------------------")
print("Matrix shape:", A.shape)
print("Number of edges:", A.sum())
print("\nFirst 10x10 section:")
print(A[:10, :10])

# TASK 1B: TRANSITION MATRIX

# Calculate the number of outgoing edges for each player
out_degree = A.sum(axis=1)

# Identify players with no outgoing edges
dangling_nodes = np.where(out_degree == 0)[0]

print("\nTASK 1B: TRANSITION MATRIX")
print("--------------------------")
print("Number of dangling nodes:", len(dangling_nodes))

# Construct the transition matrix
P = np.zeros((n, n), dtype=float)

for i in range(n):
    if out_degree[i] > 0:
        P[i, :] = A[i, :] / out_degree[i]
    else:
        # Dangling nodes can jump to any player uniformly
        P[i, :] = 1 / n

# Verify that all rows sum to one
row_sums = P.sum(axis=1)

print("Transition matrix shape:", P.shape)
print("All rows sum to 1:", np.allclose(row_sums, 1))

print("\nFirst 10x10 section:")
print(np.round(P[:10, :10], 3))

# Save the matrices for the other team members
np.savez_compressed(
    data_dir / "matrices.npz",
    A=A,
    P=P
)

print("\nMatrices saved to data/matrices.npz")

# TASK 2A: NETWORK STRUCTURE

print("\nTASK 2A: NETWORK STRUCTURE")
print("-------------------------")

# Out-degree: number of outgoing connections per player
out_degree = A.sum(axis=1)

# In-degree: number of incoming connections per player
in_degree = A.sum(axis=0)

# Create a table containing each player's degrees
player_stats = nodes[["node_id", "label", "group"]].copy()

player_stats["in_degree"] = in_degree
player_stats["out_degree"] = out_degree
player_stats["total_degree"] = in_degree + out_degree

print("\nAverage in-degree:", round(in_degree.mean(), 2))
print("Average out-degree:", round(out_degree.mean(), 2))
print("Dangling nodes:", np.sum(out_degree == 0))

print("\nTop 10 players by in-degree:")
print(
    player_stats.nlargest(10, "in_degree")[
        ["label", "group", "in_degree", "out_degree"]
    ].to_string(index=False)
)

print("\nTop 10 players by out-degree:")
print(
    player_stats.nlargest(10, "out_degree")[
        ["label", "group", "in_degree", "out_degree"]
    ].to_string(index=False)
)

# Save the statistics for later analysis
player_stats.to_csv(data_dir / "player_statistics.csv", index=False)

print("\nPlayer statistics saved successfully.")

# TASK 2B: STRONGLY CONNECTED COMPONENTS

print("\nTASK 2B: STRONGLY CONNECTED COMPONENTS")
print("--------------------------------------")

# Create a directed graph using all nodes and edges
G = nx.DiGraph()

G.add_nodes_from(nodes["node_id"])
G.add_edges_from(
    edges[["source", "target"]].itertuples(index=False, name=None)
)

# Find all strongly connected components
scc = list(nx.strongly_connected_components(G))

# Sort components by size, largest first
scc = sorted(scc, key=len, reverse=True)

print("Number of strongly connected components:", len(scc))
print("Size of largest component:", len(scc[0]))

print("\nSizes of the 10 largest components:")
print([len(component) for component in scc[:10]])

# Identify closed strongly connected components
closed_components = []

for component in scc:
    is_closed = True

    for player in component:
        for neighbor in G.successors(player):
            if neighbor not in component:
                is_closed = False
                break

        if not is_closed:
            break

    if is_closed:
        closed_components.append(component)

print("\nNumber of closed components:", len(closed_components))
print("Sizes of closed components:",
      [len(c) for c in closed_components[:10]])

# Show the players in the largest component
largest_component = scc[0]

print("\nPlayers in largest component:", len(largest_component))

# Calculate whether the network contains directed cycles
print("Network is a directed acyclic graph:", nx.is_directed_acyclic_graph(G))

# TASK 2C: NETWORK VISUALIZATIONS

figures_dir = project_dir / "figures"
figures_dir.mkdir(exist_ok=True)

# Figure 1: In-degree and out-degree distributions
plt.figure(figsize=(10, 6))

plt.hist(in_degree, bins=40, alpha=0.6,
         label="In-degree")
plt.hist(out_degree, bins=40, alpha=0.6,
         label="Out-degree")

plt.xlabel("Number of connections")
plt.ylabel("Number of players")
plt.title("ATP Tennis Network: Degree Distribution")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(figures_dir / "degree_distribution.png", dpi=300)
plt.close()

# Figure 2: Strongly connected components
component_sizes = [len(component) for component in scc]

plt.figure(figsize=(10, 6))
plt.bar(range(1, 21), component_sizes[:20])

plt.xlabel("Component rank")
plt.ylabel("Number of players")
plt.title("20 Largest Strongly Connected Components")
plt.xticks(range(1, 21))
plt.tight_layout()

plt.savefig(figures_dir / "scc_sizes.png", dpi=300)
plt.close()

print("\nTASK 2C: VISUALIZATIONS")
print("----------------------")
print("Saved figures/degree_distribution.png")
print("Saved figures/scc_sizes.png")

# TASK 2D: NETWORK GRAPH AND VALIDATION

print("\nTASK 2D: NETWORK GRAPH AND VALIDATION")
print("-------------------------------------")

# Check whether all closed components are single nodes
all_closed_singletons = all(
    len(component) == 1 for component in closed_components
)

print("All closed components have one player:", all_closed_singletons)

# Compare closed singleton nodes with dangling nodes
closed_nodes = {
    next(iter(component))
    for component in closed_components
    if len(component) == 1
}

dangling_set = set(dangling_nodes.tolist())

print("Closed components match dangling nodes:",
      all_closed_singletons and closed_nodes == dangling_set)

# Verify that all edges were correctly loaded
print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())
print("Adjacency matrix matches graph edges:",
      int(A.sum()) == G.number_of_edges())

# Visualize the largest strongly connected component
subgraph = G.subgraph(largest_component).copy()

# Use a fixed seed for reproducible layout
positions = nx.spring_layout(subgraph, seed=42)

plt.figure(figsize=(12, 10))

nx.draw_networkx_nodes(
    subgraph, positions,
    node_size=12,
    node_color="steelblue",
    alpha=0.8
)

nx.draw_networkx_edges(
    subgraph, positions,
    arrows=False,
    width=0.3,
    alpha=0.12
)

plt.title("ATP Tennis Network: Largest Strongly Connected Component")
plt.axis("off")
plt.tight_layout()

plt.savefig(figures_dir / "largest_scc_network.png", dpi=300)
plt.close()

print("Saved figures/largest_scc_network.png")
