"""
niche.py — Niche Library
=========================
Mechanism-Aware Intelligence Architecture (MAIA)

A Niche is a labeled, internalized framework for understanding a specific
invariant change-dynamic. This module manages the full lifecycle of Niches:

  - Storage:     The mechanism library as a structured collection of Niche objects
  - Selection:   Given a context, which Niche is operative?
  - Composition: When multiple Niches partially apply, how are they combined?
  - Hierarchy:   General → specific abstraction organization
  - Graph links: Similarity-based connections between related Niches

Each Niche is distinct from a feature:
  - A feature is a statistical regularity discovered to minimize loss.
  - A Niche is an invariant generative mechanism admitted by the formation gate.
  - Every Niche is a feature. Not every feature qualifies as a Niche.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
import time


# ─────────────────────────────────────────────
# Niche Data Structure
# ─────────────────────────────────────────────

@dataclass
class Niche:
    """
    A single mechanism in the library.

    Each Niche has:
      - A vector representation (the mechanism encoding)
      - A label for human-readable identification
      - Metadata tracking its lifecycle
      - Graph links to related Niches
    """
    id:             int
    vector:         torch.Tensor        # (n,) — the mechanism encoding
    label:          str                 # human-readable name
    parent_id:      Optional[int]       # parent in abstraction hierarchy
    created_at:     float               # timestamp
    activation_count: int = 0          # how many times this Niche was selected
    total_explained:  float = 0.0      # cumulative explained variance
    children:       list = field(default_factory=list)   # child Niche ids
    graph_links:    list = field(default_factory=list)   # similar Niche ids

    @property
    def norm(self) -> float:
        return self.vector.norm().item()

    @property
    def usage_rate(self) -> float:
        """How often this Niche is activated relative to its age."""
        age = max(time.time() - self.created_at, 1.0)
        return self.activation_count / age

    def __repr__(self):
        return (
            f"Niche(id={self.id}, label='{self.label}', "
            f"activations={self.activation_count}, "
            f"parent={self.parent_id})"
        )


# ─────────────────────────────────────────────
# Niche Library
# ─────────────────────────────────────────────

class NicheLibrary(nn.Module):
    def __init__(
        self,
        n_neurons: int,
        graph_link_threshold: float = 0.6,
        composition_top_k: int = 3,
    ):
        """
        Args:
            n_neurons:              Dimension of each Niche vector
            graph_link_threshold:   Cosine similarity threshold for graph linking
            composition_top_k:      Max Niches to compose for a given context
        """
        super().__init__()

        self.n = n_neurons
        self.graph_link_threshold = graph_link_threshold
        self.composition_top_k = composition_top_k

        # All Niches indexed by id
        self.niches: dict[int, Niche] = {}
        self._next_id: int = 0

        # Fast matrix access — kept in sync with niches dict
        # Columns are Niche vectors: (n, m)
        self.register_buffer("library_matrix", torch.zeros(n_neurons, 0))

        # Selection network — learns which Niche to activate given context
        # Input: context vector (n) → scores over all Niches (m)
        # Updated dynamically as Niches are added
        self.selection_weights = nn.Parameter(torch.zeros(0, n_neurons))

    @property
    def m(self) -> int:
        return len(self.niches)

    # ─────────────────────────────────────────────
    # Adding Niches
    # ─────────────────────────────────────────────

    def add_niche(
        self,
        vector: torch.Tensor,
        label: Optional[str] = None,
        parent_id: Optional[int] = None,
    ) -> Niche:
        """
        Add a new Niche to the library.
        Called by the formation gate after a candidate passes all four conditions.

        Args:
            vector:     (n,) — the orthogonalized mechanism vector
            label:      optional human-readable name
            parent_id:  parent Niche in the hierarchy (if known)

        Returns:
            The newly created Niche object
        """
        assert vector.shape == (self.n,), (
            f"Expected vector of shape ({self.n},), got {vector.shape}"
        )

        niche_id = self._next_id
        self._next_id += 1

        normalized = F.normalize(vector, dim=0)

        if label is None:
            label = f"niche_{niche_id}"

        niche = Niche(
            id=niche_id,
            vector=normalized,
            label=label,
            parent_id=parent_id,
            created_at=time.time(),
        )

        self.niches[niche_id] = niche

        # Update parent's children list
        if parent_id is not None and parent_id in self.niches:
            self.niches[parent_id].children.append(niche_id)

        # Update library matrix
        self.library_matrix = torch.cat(
            [self.library_matrix, normalized.unsqueeze(1)], dim=1
        )

        # Expand selection weights
        new_row = torch.zeros(1, self.n)
        nn.init.xavier_uniform_(new_row.unsqueeze(0))
        self.selection_weights = nn.Parameter(
            torch.cat([self.selection_weights.data, new_row], dim=0)
        )

        # Link to similar existing Niches in the graph
        self._update_graph_links(niche)

        return niche

    def _update_graph_links(self, new_niche: Niche) -> None:
        """
        Find existing Niches similar to the new one and create graph links.
        Bidirectional: both Niches get linked to each other.
        """
        if self.m < 2:
            return

        new_vec = new_niche.vector
        for nid, niche in self.niches.items():
            if nid == new_niche.id:
                continue
            sim = F.cosine_similarity(
                new_vec.unsqueeze(0),
                niche.vector.unsqueeze(0)
            ).item()
            if sim > self.graph_link_threshold:
                if nid not in new_niche.graph_links:
                    new_niche.graph_links.append(nid)
                if new_niche.id not in niche.graph_links:
                    niche.graph_links.append(new_niche.id)

    # ─────────────────────────────────────────────
    # Niche Selection
    # ─────────────────────────────────────────────

    def select(
        self,
        context: torch.Tensor,
        return_scores: bool = False,
    ) -> tuple[Niche, torch.Tensor] | Niche:
        """
        Given a context vector, select the most operative Niche.

        Selection operates on both:
          - Learned selection weights (trained affinity)
          - Direct cosine similarity to the context (structural match)

        The combination ensures selection is both learned and grounded
        in the actual geometry of the mechanism library.

        Args:
            context:        (n,) — current neuron state or change vector
            return_scores:  if True, also return raw scores

        Returns:
            The selected Niche (and optionally the score vector)
        """
        assert self.m > 0, "Cannot select from empty Niche library."
        assert context.shape == (self.n,)

        # Learned scores: (m,)
        learned_scores = self.selection_weights @ context

        # Structural scores — cosine similarity to each Niche vector: (m,)
        lib_norm = F.normalize(self.library_matrix, dim=0)  # (n, m)
        ctx_norm = F.normalize(context.unsqueeze(1), dim=0)  # (n, 1)
        structural_scores = (lib_norm * ctx_norm).sum(dim=0)  # (m,)

        # Combined score
        scores = 0.5 * learned_scores + 0.5 * structural_scores
        scores = F.softmax(scores, dim=0)

        best_idx = scores.argmax().item()
        best_niche = list(self.niches.values())[best_idx]

        # Update activation count
        best_niche.activation_count += 1

        if return_scores:
            return best_niche, scores
        return best_niche

    def compose(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, list[Niche]]:
        """
        When a context partially matches multiple Niches, compose them.

        Instead of forcing one Niche to explain everything, composition
        allows weighted combination of the top-k most relevant Niches.
        This enables handling of complex dynamics that span multiple mechanisms.

        Args:
            context: (n,) — current context vector

        Returns:
            composed_vector: (n,) — weighted combination of top-k Niche vectors
            active_niches:   list of Niches that contributed
        """
        assert self.m > 0
        assert context.shape == (self.n,)

        # Get scores for all Niches
        _, scores = self.select(context, return_scores=True)

        # Top-k Niches
        k = min(self.composition_top_k, self.m)
        top_k_scores, top_k_idx = torch.topk(scores, k)

        # Weighted sum of Niche vectors
        active_niches = []
        composed = torch.zeros(self.n)

        for score, idx in zip(top_k_scores, top_k_idx):
            niche = list(self.niches.values())[idx.item()]
            composed += score.item() * niche.vector
            active_niches.append(niche)

        composed = F.normalize(composed, dim=0)
        return composed, active_niches

    # ─────────────────────────────────────────────
    # Hierarchy Navigation
    # ─────────────────────────────────────────────

    def get_hierarchy(self, niche_id: int, depth: int = 0) -> str:
        """
        Return a string representation of the abstraction hierarchy
        rooted at the given Niche.
        """
        if niche_id not in self.niches:
            return ""

        niche = self.niches[niche_id]
        indent = "  " * depth
        lines = [f"{indent}[{niche.id}] {niche.label} "
                 f"(activations={niche.activation_count})"]

        for child_id in niche.children:
            lines.append(self.get_hierarchy(child_id, depth + 1))

        return "\n".join(lines)

    def get_root_niches(self) -> list[Niche]:
        """Return all Niches with no parent — top of the hierarchy."""
        return [n for n in self.niches.values() if n.parent_id is None]

    # ─────────────────────────────────────────────
    # Library Summary & Diagnostics
    # ─────────────────────────────────────────────

    def explain_failure(self, context: torch.Tensor) -> dict:
        """
        Given a context the model failed on, diagnose why.

        Maps directly to the three failure modes from the spec:
        1. Incomplete mechanism — no Niche covers this context well
        2. Wrong selection      — selection scores are ambiguous
        3. Insufficient awareness — m is too small

        Returns a dict with failure attribution.
        """
        if self.m == 0:
            return {"failure": "insufficient_awareness", "m": 0,
                "detail": "No Niches in library yet."}

        _, scores = self.select(context, return_scores=True)
        best_score = scores.max().item()
        entropy = -(scores * (scores + 1e-8).log()).sum().item()

        uniform_score = 1.0 / self.m
        match_ratio = best_score / uniform_score

        if match_ratio < 1.2:
            return {
                "failure": "incomplete_mechanism",
                "best_score": best_score,
                "match_ratio": match_ratio,
                "detail": "No existing Niche matches this context well. "
                      "Residual should be passed to formation gate."
            }
        elif entropy > 1.5:
            return {
                "failure": "wrong_selection",
                "entropy": entropy,
                "detail": "Selection scores are highly uncertain. "
                      "Multiple Niches compete — composition may help."
            }
        else:
            return {
                "failure": "none_detected",
                "best_score": best_score,
                "match_ratio": match_ratio,
                "detail": "Mechanism coverage looks adequate for this context."
            }

    def summary(self) -> str:
        lines = [
            f"NicheLibrary Summary",
            f"  Active Niches:     {self.m}",
            f"  Root Niches:       {len(self.get_root_niches())}",
        ]
        if self.m > 0:
            most_used = max(self.niches.values(), key=lambda n: n.activation_count)
            least_used = min(self.niches.values(), key=lambda n: n.activation_count)
            lines += [
                f"  Most used:         [{most_used.id}] {most_used.label} "
                f"({most_used.activation_count} activations)",
                f"  Least used:        [{least_used.id}] {least_used.label} "
                f"({least_used.activation_count} activations)",
            ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    lib = NicheLibrary(n_neurons=32, composition_top_k=3)

    print("NicheLibrary — Demonstration")
    print("=" * 45)

    # Add some Niches manually (in real use, formation_gate admits these)
    vectors = [
        ("rate_of_change",      torch.randn(32)),
        ("cyclic_pattern",      torch.randn(32)),
        ("causal_drift",        torch.randn(32)),
        ("phase_transition",    torch.randn(32)),
        ("relational_change",   torch.randn(32)),
    ]

    for label, vec in vectors:
        niche = lib.add_niche(vec, label=label)
        print(f"Added: {niche}")

    print(f"\n{lib.summary()}")

    # Test selection
    print("\n--- Selection ---")
    context = torch.randn(32)
    selected, scores = lib.select(context, return_scores=True)
    print(f"Context → selected Niche: [{selected.id}] {selected.label}")
    print(f"Score distribution: {scores.detach().numpy().round(3)}")

    # Test composition
    print("\n--- Composition ---")
    composed, active = lib.compose(context)
    print(f"Active Niches in composition:")
    for n in active:
        print(f"  [{n.id}] {n.label}")

    # Test hierarchy
    print("\n--- Hierarchy ---")
    child = lib.add_niche(torch.randn(32), label="rate_subtype_A", parent_id=0)
    print(lib.get_hierarchy(0))

    # Test failure explanation
    print("\n--- Failure Diagnosis ---")
    unknown_context = torch.randn(32) * 5  # very different from training
    diagnosis = lib.explain_failure(unknown_context)
    print(f"Failure type:  {diagnosis['failure']}")
    print(f"Detail:        {diagnosis['detail']}")