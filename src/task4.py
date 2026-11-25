"""Task 4: verify the stationary distribution and explain convergence."""

import numpy as np

from task3 import load_network, run_from_starting_players


def stationary_distribution(transition):
    """Find and normalize the eigenvector of P.T associated with eigenvalue 1."""
    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    at_one = np.isclose(eigenvalues, 1.0, atol=1e-8, rtol=0)
    if at_one.sum() != 1:
        raise ValueError("Expected one eigenvalue at 1 for the completed ATP walk.")

    candidate = np.real_if_close(eigenvectors[:, np.flatnonzero(at_one)[0]])
    if np.iscomplexobj(candidate) or abs(candidate.sum()) < 1e-12:
        raise ValueError("The stationary candidate must be real and normalizable.")

    # Normalizing by the sum also removes the eigenvector's arbitrary sign.
    stationary = candidate / candidate.sum()
    residual = stationary @ transition - stationary
    if (
        np.any(stationary < 0)
        or not np.isclose(stationary.sum(), 1, atol=1e-12, rtol=0)
        or not np.linalg.norm(residual, ord=1) < 1e-12
    ):
        raise ValueError("The candidate failed normalization, nonnegativity, or stationarity.")
    return stationary, eigenvalues, at_one, residual


def reachable(support, start):
    """Find all players reachable by following the directed arrows in support."""
    seen = {start}
    pending = [start]
    while pending:
        for neighbor in np.flatnonzero(support[pending.pop()]):
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return seen


def print_stationary_equations(adjacency, stationary, player_ids, names_by_id):
    """Expand the incoming contributions for Bopanna and Damm."""
    out_degree = adjacency.sum(axis=1)
    uniform_term = stationary[out_degree == 0].sum() / len(stationary)
    for name in ("Rohan Bopanna", "Martin Damm"):
        target = player_ids[name]
        sources = np.flatnonzero(adjacency[:, target])
        terms = stationary[sources] / out_degree[sources]
        total = terms.sum() + uniform_term
        print(f"\nStationary equation for {name}:")
        for source, term in zip(sources, terms):
            print(
                f"  {names_by_id[source]}: "
                f"{stationary[source]:.9e} / {out_degree[source]} = {term:.9e}"
            )
        print(f"  Uniform contribution from dangling players: {uniform_term:.9e}")
        print(f"  Sum: {total:.9e}  |  Eigenvector entry: {stationary[target]:.9e}")
        np.testing.assert_allclose(total, stationary[target], atol=1e-14, rtol=0)


def print_graph_evidence(adjacency, transition, player_ids):
    """Check irreducibility and exhibit return paths of coprime lengths."""
    support = transition > 0
    djokovic = player_ids["Novak Djokovic"]
    forward = reachable(support, djokovic)
    backward = reachable(support.T, djokovic)
    print(f"\nDjokovic reaches {len(forward)}/{len(support)} players in P.")
    print(f"{len(backward)}/{len(support)} players can reach Djokovic in P.")
    if len(forward) != len(support) or len(backward) != len(support):
        raise ValueError("The completed walk is not strongly connected via Djokovic.")

    cycles = (
        ("Novak Djokovic", "Rafael Nadal", "Novak Djokovic"),
        ("Novak Djokovic", "Alexander Zverev", "Roger Federer", "Novak Djokovic"),
    )
    for cycle in cycles:
        ids = [player_ids[name] for name in cycle]
        if not all(adjacency[source, target] for source, target in zip(ids, ids[1:])):
            raise ValueError("The proposed return path is missing a match edge.")
        print(f"{len(ids) - 1}-step return: " + " -> ".join(cycle))
    print("One closed class in P contains every player, so the stationary distribution is unique.")
    print("The 2- and 3-step returns have gcd 1, so every starting distribution converges.")


def print_stationary_summary(stationary, residual):
    """Report the probability checks and the size of the stationary residual."""
    print(f"Stationary sum: {stationary.sum():.12f}")
    print(f"Minimum probability: {stationary.min():.9e}")
    print(f"L1 stationary residual: {np.linalg.norm(residual, ord=1):.3e}")


def print_walk_agreement(stationary, runs):
    """Compare the eigenvector result to the final distributions from Task 3."""
    for name, result in runs.items():
        print(f"L1 distance to Task 3's {name} result: {np.abs(result.final - stationary).sum():.3e}")


def print_top_players(stationary, names_by_id):
    """Report the five highest stationary probabilities."""
    print("\nTop five players in the original walk:")
    for rank, index in enumerate(np.argsort(-stationary)[:5], start=1):
        print(f"{rank}. {names_by_id[index]}: {stationary[index]:.6%}")


def print_spectrum(eigenvalues, at_one):
    """Report the dominant eigenvalues and check for an alternating mode."""
    print("\nLeading eigenvalues by modulus:")
    for value in eigenvalues[np.argsort(-np.abs(eigenvalues))[:5]]:
        print(f"  {value.real:+.8f} {value.imag:+.8f}i  |lambda| = {abs(value):.8f}")
    print(f"Largest modulus other than 1: {np.abs(eigenvalues[~at_one]).max():.8f}")
    near_minus_one = np.isclose(eigenvalues, -1, atol=1e-8, rtol=0)
    print(f"Eigenvalues near -1: {near_minus_one.sum()}")


def main():
    adjacency, transition, player_ids = load_network()
    names_by_id = {index: name for name, index in player_ids.items()}
    stationary, eigenvalues, at_one, residual = stationary_distribution(transition)
    print_stationary_summary(stationary, residual)
    print_walk_agreement(stationary, run_from_starting_players(transition, player_ids))
    print_top_players(stationary, names_by_id)
    print_stationary_equations(adjacency, stationary, player_ids, names_by_id)
    print_graph_evidence(adjacency, transition, player_ids)
    print_spectrum(eigenvalues, at_one)


if __name__ == "__main__":
    main()
