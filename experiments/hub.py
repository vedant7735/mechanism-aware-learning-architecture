"""
hub.py — Central Awareness Hub
================================
Mechanism-Aware Intelligence Architecture (MAIA)

The Hub is the coordination core of MAIA. It operates as a dual-signal system:

  Signal 1 — Change Vector:
    Tracks what changed per neuron, by how much, and why.
    Each neuron's change is decomposed into:
      - contributions from existing Niche activations (explained change)
      - a residual (unexplained change → candidate for new Niche formation)

  Signal 2 — Sparse Correlation Structure:
    Tracks which neurons change together.
    Not a full nxn matrix — a sparse approximation of the most significant
    co-change relationships.

The Hub's primary function is weight modulation:
  - It adjusts the mechanism-factorized state based on these two signals.
  - It does NOT minimize a loss directly.
  - It does NOT generate outputs.
  - It makes the model's mechanism understanding richer over time.
"""

import torch
import torch.nn as nn
from typing import Optional


class CentralAwarenessHub(nn.Module):
    def __init__(
        self,
        n_neurons: int,
        n_niches: int,
        top_k_correlations: int = 10,
        residual_threshold: float = 0.1,
    ):
        """
        Args:
            n_neurons:            Number of neurons / local pattern processors (n)
            n_niches:             Number of active Niche columns (m)
            top_k_correlations:   How many top co-change pairs to track (sparse approx)
            residual_threshold:   Minimum residual magnitude to flag as candidate Niche
        """
        super().__init__()

        self.n = n_neurons
        self.m = n_niches
        self.top_k = top_k_correlations
        self.residual_threshold = residual_threshold

        # Mechanism-factorized state: n neurons × m Niches
        # Each column is a Niche vector encoding an invariant change-dynamic
        self.mechanism_state = nn.Parameter(
            torch.randn(n_neurons, n_niches) * 0.01
        )

        # Weight modulation layer: maps hub signals → state update
        # Input: change vector (n) + flattened sparse correlations (top_k * 3)
        signal_dim = n_neurons + top_k_correlations * 3  # (i, j, strength) per pair
        self.modulation_layer = nn.Sequential(
            nn.Linear(signal_dim, n_neurons * n_niches),
            nn.Tanh()
        )

        # Previous neuron state — needed to compute change vector
        self.register_buffer("prev_state", torch.zeros(n_neurons))

        # Running attribution matrix: how much each Niche explains each neuron's change
        self.register_buffer(
            "attribution", torch.zeros(n_neurons, n_niches)
        )

    # ─────────────────────────────────────────────
    # Signal 1: Change Vector
    # ─────────────────────────────────────────────

    def compute_change_vector(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Compute per-neuron change from previous state.

        Returns:
            change_vector: (n,) — how much each neuron changed this step
        """
        change_vector = current_state - self.prev_state
        self.prev_state = current_state.detach()
        return change_vector

    def decompose_change(
        self,
        change_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose the change vector into:
          - explained: contributions from existing Niche activations
          - residual:  unexplained change → candidate for new Niche

        This is the attribution step. Each Niche column in mechanism_state
        explains some portion of the observed neuron changes.

        Returns:
            explained:  (n,) — change attributed to existing Niches
            residual:   (n,) — change not explained by any current Niche
        """
        # Project change vector onto each Niche column
        # niche_activations: (m,) — how active each Niche is given this change
        niche_activations = self.mechanism_state.T @ change_vector  # (m,)

        # Reconstruct the explained portion
        explained = self.mechanism_state @ niche_activations  # (n,)

        # Residual is what no Niche explains
        residual = change_vector - explained  # (n,)

        # Update running attribution
        self.attribution = (
            0.9 * self.attribution
            + 0.1 * torch.outer(change_vector.abs(), niche_activations.abs())
        )

        return explained, residual

    def flag_candidate_niche(self, residual: torch.Tensor) -> bool:
        """
        Check if the residual is large enough to warrant Niche formation.
        Passes the residual to formation_gate.py for full invariance checks.

        Returns:
            True if residual magnitude exceeds threshold (candidate flagged)
        """
        return residual.norm().item() > self.residual_threshold

    # ─────────────────────────────────────────────
    # Signal 2: Sparse Correlation Structure
    # ─────────────────────────────────────────────

    def compute_sparse_correlations(
        self, change_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute which neurons change together — sparse approximation.

        Instead of a full n×n matrix (quadratic), we track only the
        top-k most significant co-change pairs.

        Returns:
            sparse_signal: (top_k * 3,) — flattened (i, j, strength) triplets
        """
        n = self.n
        change = change_vector.detach()

        # Outer product gives full co-change matrix — we only keep top-k entries
        co_change = torch.outer(change, change)  # (n, n)

        # Zero out diagonal (self-correlation)
        co_change.fill_diagonal_(0)

        # Take absolute value and find top-k pairs
        abs_co = co_change.abs()
        flat = abs_co.view(-1)
        top_k_vals, top_k_idx = torch.topk(flat, min(self.top_k, flat.numel()))

        # Convert flat indices back to (i, j) pairs
        rows = (top_k_idx // n).float()
        cols = (top_k_idx % n).float()

        # Triplets: (i, j, strength) — normalized
        max_val = top_k_vals.max().clamp(min=1e-8)
        strengths = top_k_vals / max_val

        # Pad to fixed size if fewer than top_k pairs found
        triplets = torch.stack([rows, cols, strengths], dim=1).view(-1)
        pad_size = self.top_k * 3 - triplets.numel()
        if pad_size > 0:
            triplets = torch.cat([triplets, torch.zeros(pad_size)])

        return triplets[:self.top_k * 3]

    # ─────────────────────────────────────────────
    # Weight Modulation
    # ─────────────────────────────────────────────

    def modulate_weights(
        self,
        change_vector: torch.Tensor,
        sparse_signal: torch.Tensor,
    ) -> None:
        """
        Adjust the mechanism-factorized state based on both hub signals.

        This is the Hub's primary function — not loss minimization,
        but structured state update driven by observed change dynamics.
        """
        # Concatenate both signals
        hub_signal = torch.cat([change_vector, sparse_signal])  # (n + top_k*3,)

        # Compute state delta via modulation layer
        delta = self.modulation_layer(hub_signal)  # (n * m,)
        delta = delta.view(self.n, self.m)  # reshape to state dimensions

        # Apply update with small learning rate to mechanism state
        with torch.no_grad():
            self.mechanism_state += 0.01 * delta

    # ─────────────────────────────────────────────
    # Main Step
    # ─────────────────────────────────────────────

    def step(
        self,
        current_neuron_state: torch.Tensor,
    ) -> dict:
        """
        Full Hub update step given current neuron states.

        Args:
            current_neuron_state: (n,) — current activation of all neurons

        Returns:
            A dict containing all signals and flags for this step:
              - change_vector:   (n,)
              - explained:       (n,)
              - residual:        (n,)
              - sparse_signal:   (top_k * 3,)
              - candidate_niche: bool — whether residual flags a new Niche candidate
              - niche_activations: (m,) — how active each Niche was this step
        """
        assert current_neuron_state.shape == (self.n,), (
            f"Expected neuron state of shape ({self.n},), "
            f"got {current_neuron_state.shape}"
        )

        # Signal 1: change vector + decomposition
        change_vector = self.compute_change_vector(current_neuron_state)
        explained, residual = self.decompose_change(change_vector)
        candidate = self.flag_candidate_niche(residual)

        # Signal 2: sparse correlations
        sparse_signal = self.compute_sparse_correlations(change_vector)

        # Niche activations for this step
        niche_activations = self.mechanism_state.T @ change_vector  # (m,)

        # Modulate weights
        self.modulate_weights(change_vector, sparse_signal)

        return {
            "change_vector": change_vector,
            "explained": explained,
            "residual": residual,
            "sparse_signal": sparse_signal,
            "candidate_niche": candidate,
            "niche_activations": niche_activations,
        }

    # ─────────────────────────────────────────────
    # Introspection
    # ─────────────────────────────────────────────

    def get_attribution_summary(self) -> dict:
        """
        Returns which Niches are most responsible for explaining neuron changes.
        Useful for debugging, visualization, and failure tracing.
        """
        niche_importance = self.attribution.sum(dim=0)  # (m,) — per Niche
        neuron_coverage = self.attribution.sum(dim=1)   # (n,) — per neuron

        return {
            "niche_importance": niche_importance,
            "neuron_coverage": neuron_coverage,
            "most_active_niche": niche_importance.argmax().item(),
            "least_explained_neuron": neuron_coverage.argmin().item(),
        }

    def __repr__(self):
        return (
            f"CentralAwarenessHub("
            f"n_neurons={self.n}, "
            f"n_niches={self.m}, "
            f"top_k_correlations={self.top_k})"
        )


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    hub = CentralAwarenessHub(
        n_neurons=32,
        n_niches=8,
        top_k_correlations=10,
        residual_threshold=0.1,
    )

    print(hub)
    print(f"\nMechanism state shape: {hub.mechanism_state.shape}")

    # Simulate a few steps with random neuron states
    for step_idx in range(5):
        neuron_state = torch.randn(32)
        result = hub.step(neuron_state)

        print(f"\nStep {step_idx + 1}:")
        print(f"  Change vector norm:     {result['change_vector'].norm():.4f}")
        print(f"  Explained norm:         {result['explained'].norm():.4f}")
        print(f"  Residual norm:          {result['residual'].norm():.4f}")
        print(f"  Candidate Niche:        {result['candidate_niche']}")
        print(f"  Top Niche activation:   {result['niche_activations'].abs().argmax().item()}")

    summary = hub.get_attribution_summary()
    print(f"\nAttribution Summary:")
    print(f"  Most active Niche:        {summary['most_active_niche']}")
    print(f"  Least explained neuron:   {summary['least_explained_neuron']}")