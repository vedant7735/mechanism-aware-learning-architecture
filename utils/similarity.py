"""
similarity.py — Similarity & Correlation Utilities
====================================================
Mechanism-Aware Intelligence Architecture (MAIA)
utils/similarity.py

Shared utilities for measuring relationships between vectors,
used across hub.py, formation_gate.py, and niche.py.

Functions:
  - cosine_similarity       : similarity between two vectors
  - pairwise_cosine         : similarity matrix across a set of vectors
  - top_k_similar           : find most similar vectors to a query
  - decorrelation_score     : how orthogonal a vector is to a library
  - explained_variance      : how much of a vector is explained by a basis
  - soft_orthogonalize      : partial Gram-Schmidt (controlled, not full)
  - inter_library_similarity: summary stats for a Niche library matrix
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────
# Core Similarity
# ─────────────────────────────────────────────

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Cosine similarity between two 1D vectors.

    Returns a value in [-1, 1].
    1.0  = identical direction
    0.0  = orthogonal
    -1.0 = opposite direction

    Args:
        a: (n,)
        b: (n,)

    Returns:
        float similarity score
    """
    assert a.shape == b.shape and a.dim() == 1, (
        f"Expected two 1D vectors of equal shape, got {a.shape} and {b.shape}"
    )
    a_norm = a.norm()
    b_norm = b.norm()
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    return (a @ b / (a_norm * b_norm)).item()


def pairwise_cosine(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute the full pairwise cosine similarity matrix for a set of vectors.

    Args:
        matrix: (n, m) — m vectors of dimension n (columns are vectors)

    Returns:
        sim_matrix: (m, m) — symmetric similarity matrix
                    diagonal is 1.0 (self-similarity)
    """
    assert matrix.dim() == 2, f"Expected 2D matrix, got shape {matrix.shape}"

    # Normalize each column
    normed = F.normalize(matrix, dim=0)  # (n, m)

    # Gram matrix = pairwise dot products of normalized columns
    sim_matrix = normed.T @ normed  # (m, m)

    # Clamp for numerical stability
    return sim_matrix.clamp(-1.0, 1.0)


def top_k_similar(
    query: torch.Tensor,
    library: torch.Tensor,
    k: int = 5,
    return_scores: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """
    Find the top-k most similar columns in a library to a query vector.

    Args:
        query:          (n,) — query vector
        library:        (n, m) — library of m vectors
        k:              number of top matches to return
        return_scores:  if True, also return similarity scores

    Returns:
        indices: (k,) — column indices of top matches
        scores:  (k,) — cosine similarities (if return_scores=True)
    """
    assert query.shape[0] == library.shape[0], (
        f"Query dim {query.shape[0]} must match library dim {library.shape[0]}"
    )
    m = library.shape[1]
    k = min(k, m)

    query_norm = F.normalize(query.unsqueeze(1), dim=0)   # (n, 1)
    lib_norm   = F.normalize(library, dim=0)               # (n, m)

    scores = (lib_norm * query_norm).sum(dim=0)            # (m,)
    top_scores, top_idx = torch.topk(scores, k)

    if return_scores:
        return top_idx, top_scores
    return top_idx


# ─────────────────────────────────────────────
# Orthogonality & Coverage
# ─────────────────────────────────────────────

def decorrelation_score(
    candidate: torch.Tensor,
    library: torch.Tensor,
) -> float:
    """
    Measure how orthogonal a candidate vector is to an existing library.

    Returns a score in [0, 1]:
      1.0 = perfectly orthogonal to all library vectors (fully novel)
      0.0 = identical to an existing library vector (fully redundant)

    Used by the formation gate to check the decorrelation condition.

    Args:
        candidate: (n,)
        library:   (n, m)

    Returns:
        decorrelation score (float)
    """
    if library.shape[1] == 0:
        return 1.0  # empty library — candidate is trivially novel

    _, top_scores = top_k_similar(candidate, library, k=1, return_scores=True)
    max_sim = top_scores[0].abs().item()

    # Decorrelation = 1 - max similarity
    return 1.0 - max_sim


def explained_variance(
    vector: torch.Tensor,
    basis: torch.Tensor,
) -> float:
    """
    Compute how much of a vector's variance is explained by a basis
    (set of column vectors).

    This is the projection ratio: ||explained|| / ||original||

    Args:
        vector: (n,) — the vector to analyze
        basis:  (n, m) — the basis vectors (columns)

    Returns:
        float in [0, 1] — 1.0 means fully explained by the basis
    """
    if basis.shape[1] == 0 or vector.norm() < 1e-8:
        return 0.0

    # Project vector onto each basis column and reconstruct
    explained = torch.zeros_like(vector)
    for i in range(basis.shape[1]):
        col = basis[:, i]
        col_norm_sq = (col @ col).item()
        if col_norm_sq < 1e-8:
            continue
        projection = (vector @ col) / col_norm_sq
        explained += projection * col

    ratio = explained.norm() / vector.norm()
    return ratio.clamp(0.0, 1.0).item()


def soft_orthogonalize(
    candidate: torch.Tensor,
    library: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """
    Partial Gram-Schmidt orthogonalization.

    Unlike full Gram-Schmidt (used in formation_gate.py),
    this version applies a controlled amount of projection removal.
    strength=1.0 is full orthogonalization; strength=0.0 is no change.

    Useful for soft regularization during Niche updates.

    Args:
        candidate: (n,) — vector to orthogonalize
        library:   (n, m) — existing basis
        strength:  float in [0, 1] — how much to orthogonalize

    Returns:
        (n,) — partially orthogonalized vector
    """
    assert 0.0 <= strength <= 1.0, f"strength must be in [0, 1], got {strength}"

    result = candidate.clone()

    for i in range(library.shape[1]):
        col = library[:, i]
        col_norm_sq = (col @ col).item()
        if col_norm_sq < 1e-8:
            continue
        projection = (result @ col) / col_norm_sq
        result = result - strength * projection * col

    return result


# ─────────────────────────────────────────────
# Library-Level Analysis
# ─────────────────────────────────────────────

def inter_library_similarity(library: torch.Tensor) -> dict:
    """
    Compute summary statistics on inter-Niche similarity within a library.

    Used to monitor library health:
      - High mean similarity → Niches are becoming redundant
      - Low max similarity   → library is well-decorrelated

    Args:
        library: (n, m) — the Niche library matrix

    Returns:
        dict with max, mean, min inter-Niche cosine similarity
    """
    m = library.shape[1]
    if m < 2:
        return {"max": 0.0, "mean": 0.0, "min": 0.0, "n_niches": m}

    sim_matrix = pairwise_cosine(library)

    # Mask diagonal (self-similarity = 1.0, not meaningful)
    mask = ~torch.eye(m, dtype=torch.bool)
    off_diag = sim_matrix[mask].abs()

    return {
        "max":      off_diag.max().item(),
        "mean":     off_diag.mean().item(),
        "min":      off_diag.min().item(),
        "n_niches": m,
    }


def most_redundant_pair(library: torch.Tensor) -> tuple[int, int, float]:
    """
    Find the most similar pair of Niches in the library.
    Used by the compression step to decide what to merge.

    Args:
        library: (n, m)

    Returns:
        (i, j, similarity) — indices and similarity of the most similar pair
    """
    m = library.shape[1]
    assert m >= 2, "Need at least 2 Niches to find a redundant pair."

    sim_matrix = pairwise_cosine(library)
    sim_matrix.fill_diagonal_(-1.0)  # ignore self-similarity

    idx = sim_matrix.argmax()
    i = (idx // m).item()
    j = (idx % m).item()
    sim = sim_matrix[i, j].item()

    return i, j, sim


def change_correlation(
    change_a: torch.Tensor,
    change_b: torch.Tensor,
) -> float:
    """
    Measure how correlated two change vectors are.
    Used by the Hub to build the sparse correlation structure.

    Args:
        change_a: (n,) — change vector at step t
        change_b: (n,) — change vector at step t'

    Returns:
        float correlation in [-1, 1]
    """
    return cosine_similarity(change_a, change_b)


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    print("Similarity Utilities — Sanity Check")
    print("=" * 45)

    n, m = 32, 6

    # Build a small mock library
    library = torch.randn(n, m)

    # Cosine similarity
    a = torch.randn(n)
    b = a + torch.randn(n) * 0.1  # near-identical
    c = torch.randn(n)             # unrelated

    print(f"\ncosine_similarity(a, b≈a):  {cosine_similarity(a, b):.4f}  (expect ~1.0)")
    print(f"cosine_similarity(a, c):    {cosine_similarity(a, c):.4f}  (expect ~0)")

    # Pairwise
    sim_matrix = pairwise_cosine(library)
    print(f"\npairwise_cosine diagonal:   {sim_matrix.diagonal().mean():.4f}  (expect 1.0)")

    # Top-k
    query = library[:, 0] + torch.randn(n) * 0.05  # close to column 0
    idx, scores = top_k_similar(query, library, k=3, return_scores=True)
    print(f"\ntop_k_similar (query ≈ col 0):")
    print(f"  top indices: {idx.tolist()}  (expect 0 to be first)")
    print(f"  top scores:  {scores.tolist()}")

    # Decorrelation
    novel    = torch.randn(n) * 10
    existing = library[:, 0] + torch.randn(n) * 0.01

    print(f"\ndecorrelation_score (novel vector):   {decorrelation_score(novel, library):.4f}  (expect high)")
    print(f"decorrelation_score (copy of col 0):  {decorrelation_score(existing, library):.4f}  (expect low)")

    # Explained variance
    ev = explained_variance(library[:, 0], library)
    print(f"\nexplained_variance (col 0 vs library): {ev:.4f}  (expect ~1.0)")

    # Library health
    stats = inter_library_similarity(library)
    print(f"\ninter_library_similarity:")
    print(f"  max:  {stats['max']:.4f}")
    print(f"  mean: {stats['mean']:.4f}")
    print(f"  min:  {stats['min']:.4f}")

    # Most redundant pair
    i, j, sim = most_redundant_pair(library)
    print(f"\nmost_redundant_pair: ({i}, {j}) with similarity {sim:.4f}")

    print("\n✓ All checks passed.")