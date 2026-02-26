"""
main.py — MAIA System
======================
Mechanism-Aware Intelligence Architecture

Wires together:
  - CentralAwarenessHub    (hub.py)
  - NicheFormationGate     (formation_gate.py)
  - NicheLibrary           (niche.py)

The full loop:

  1. Receive neuron state from environment
  2. Hub computes change vector + sparse correlations
  3. Hub decomposes change into explained + residual
  4. If residual is large → pass to formation gate
  5. If gate admits → add to Niche library
  6. Library selects / composes operative Niche for current context
  7. Hub modulates weights based on both signals
  8. Repeat

This is the minimal end-to-end MAIA system.
No task head yet — this is the Foundation Model core.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from hub import CentralAwarenessHub
from experiments.formation_gate import NicheFormationGate
from niche import NicheLibrary, Niche


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class MAIAConfig:
    n_neurons:              int   = 64      # neuron count
    max_niches:             int   = 32      # max Niches before compression
    top_k_correlations:     int   = 16      # sparse correlation pairs
    residual_threshold:     float = 0.08    # Hub: min residual to flag candidate
    similarity_threshold:   float = 0.85   # Gate: max cosine sim to existing Niches
    coverage_threshold:     float = 0.05   # Gate: min residual norm after projection
    stability_threshold:    float = 0.25   # Gate: min stability score
    history_length:         int   = 12     # Gate: candidate history window
    graph_link_threshold:   float = 0.6    # Library: similarity for graph links
    composition_top_k:      int   = 3      # Library: max Niches to compose


# ─────────────────────────────────────────────
# MAIA Core
# ─────────────────────────────────────────────

class MAIA(nn.Module):
    def __init__(self, config: MAIAConfig):
        super().__init__()
        self.config = config
        self.n = config.n_neurons

        # Core components
        self.hub = CentralAwarenessHub(
            n_neurons=config.n_neurons,
            n_niches=max(1, config.max_niches // 4),  # start small, grows with library
            top_k_correlations=config.top_k_correlations,
            residual_threshold=config.residual_threshold,
        )

        self.gate = NicheFormationGate(
            n_neurons=config.n_neurons,
            max_niches=config.max_niches,
            similarity_threshold=config.similarity_threshold,
            coverage_threshold=config.coverage_threshold,
            stability_threshold=config.stability_threshold,
            history_length=config.history_length,
        )

        self.library = NicheLibrary(
            n_neurons=config.n_neurons,
            graph_link_threshold=config.graph_link_threshold,
            composition_top_k=config.composition_top_k,
        )

        # Step counter
        self.step_count = 0

        # System log
        self.log: list[dict] = []

    # ─────────────────────────────────────────────
    # Core Step
    # ─────────────────────────────────────────────

    def step(self, neuron_state: torch.Tensor) -> dict:
        """
        Full MAIA update step.

        Args:
            neuron_state: (n,) — current neuron activations from environment

        Returns:
            step_info dict with all signals, selections, and events
        """
        assert neuron_state.shape == (self.n,), (
            f"Expected neuron state of shape ({self.n},), got {neuron_state.shape}"
        )

        self.step_count += 1
        step_info = {"step": self.step_count}

        # ── 1. Hub: compute signals ──────────────────────
        hub_result = self.hub.step(neuron_state)

        step_info["change_norm"]    = hub_result["change_vector"].norm().item()
        step_info["explained_norm"] = hub_result["explained"].norm().item()
        step_info["residual_norm"]  = hub_result["residual"].norm().item()
        step_info["candidate_flagged"] = hub_result["candidate_niche"]

        # ── 2. Formation gate: evaluate candidate ────────
        new_niche_formed = False
        gate_result = None

        if hub_result["candidate_niche"]:
            residual = hub_result["residual"]
            gate_result = self.gate.evaluate(residual)
            step_info["gate_admitted"] = gate_result.admitted
            step_info["gate_reason"]   = gate_result.failure_reason

            if gate_result.admitted:
                # Add to library
                new_niche = self.library.add_niche(
                    gate_result.orthogonal_vector,
                    label=f"niche_{self.library.m}",
                )
                new_niche_formed = True
                step_info["new_niche_id"]    = new_niche.id
                step_info["new_niche_label"] = new_niche.label
        else:
            step_info["gate_admitted"] = None
            step_info["gate_reason"]   = None

        step_info["new_niche_formed"] = new_niche_formed
        step_info["library_size"]     = self.library.m

        # ── 3. Library: select / compose operative Niche ─
        if self.library.m > 0:
            if self.library.m >= self.config.composition_top_k:
                # Enough Niches to compose
                composed, active_niches = self.library.compose(neuron_state)
                step_info["selection_mode"]   = "composition"
                step_info["active_niches"]    = [n.id for n in active_niches]
                step_info["operative_vector"] = composed
            else:
                # Single selection
                selected, scores = self.library.select(neuron_state, return_scores=True)
                step_info["selection_mode"]   = "single"
                step_info["active_niches"]    = [selected.id]
                step_info["operative_vector"] = selected.vector
        else:
            step_info["selection_mode"]   = "none"
            step_info["active_niches"]    = []
            step_info["operative_vector"] = None

        self.log.append(step_info)
        return step_info

    # ─────────────────────────────────────────────
    # Failure Diagnosis
    # ─────────────────────────────────────────────

    def diagnose(self, context: torch.Tensor) -> dict:
        """
        Given a context the system failed on, explain why.
        Delegates to library's explain_failure method.
        """
        return self.library.explain_failure(context)

    # ─────────────────────────────────────────────
    # System Summary
    # ─────────────────────────────────────────────

    def summary(self) -> str:
        total_candidates = sum(
            1 for s in self.log if s.get("candidate_flagged")
        )
        total_admitted = sum(
            1 for s in self.log if s.get("new_niche_formed")
        )
        total_rejected = total_candidates - total_admitted

        hub_summary   = self.hub.get_attribution_summary()
        gate_summary  = self.gate.get_library_summary()

        lines = [
            "=" * 50,
            "MAIA System Summary",
            "=" * 50,
            f"  Steps run:             {self.step_count}",
            f"  Active Niches:         {self.library.m}",
            f"  Candidates flagged:    {total_candidates}",
            f"  Niches admitted:       {total_admitted}",
            f"  Candidates rejected:   {total_rejected}",
            f"  Most active Niche:     {hub_summary['most_active_niche']}",
            f"  Least explained neuron:{hub_summary['least_explained_neuron']}",
            "",
            self.library.summary(),
            "=" * 50,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Demo: Toy Dynamics
# ─────────────────────────────────────────────

def generate_neuron_state(step: int, n: int, mode: str = "mixed") -> torch.Tensor:
    """
    Generate synthetic neuron states with different underlying dynamics.
    Each mode represents a different change-mechanism — used to test
    whether MAIA discovers and reuses distinct Niches for each.
    """
    t = step * 0.1

    if mode == "linear":
        # Steady linear drift — one clear mechanism
        base = torch.linspace(0, 1, n) * t * 0.1
        return base + torch.randn(n) * 0.01

    elif mode == "oscillatory":
        # Cyclic pattern — different mechanism
        base = torch.sin(torch.linspace(0, 3.14, n) * t)
        return base + torch.randn(n) * 0.01

    elif mode == "jump":
        # Sudden phase transition every 20 steps
        phase = (step // 20) % 2
        base = torch.ones(n) * phase
        base[:n//2] *= -1 if phase else 1
        return base + torch.randn(n) * 0.02

    elif mode == "mixed":
        # Combination — tests composition
        linear = torch.linspace(0, 1, n) * t * 0.05
        osc    = torch.sin(torch.linspace(0, 3.14, n) * t) * 0.3
        return linear + osc + torch.randn(n) * 0.01

    else:
        return torch.randn(n)


def run_demo():
    torch.manual_seed(42)

    config = MAIAConfig(
        n_neurons=32,
        max_niches=16,
        residual_threshold=0.05,
        stability_threshold=0.15,   # lower for demo — more Niche formation
        coverage_threshold=0.03,
        history_length=6,
    )

    maia = MAIA(config)

    print("MAIA — End-to-End Demonstration")
    print("=" * 50)
    print(f"Neurons: {config.n_neurons}  |  Max Niches: {config.max_niches}")
    print()

    # Phase 1: Linear dynamics — MAIA should form a Niche for this
    print("Phase 1: Linear dynamics (steps 1–30)")
    for step in range(1, 31):
        state = generate_neuron_state(step, config.n_neurons, mode="linear")
        info = maia.step(state)
        if info["new_niche_formed"]:
            print(f"  Step {step:3d}: ✓ New Niche formed → "
                  f"[{info['new_niche_id']}] {info['new_niche_label']}  "
                  f"(library size: {info['library_size']})")

    # Phase 2: Oscillatory dynamics — should form a different Niche
    print("\nPhase 2: Oscillatory dynamics (steps 31–60)")
    for step in range(31, 61):
        state = generate_neuron_state(step, config.n_neurons, mode="oscillatory")
        info = maia.step(state)
        if info["new_niche_formed"]:
            print(f"  Step {step:3d}: ✓ New Niche formed → "
                  f"[{info['new_niche_id']}] {info['new_niche_label']}  "
                  f"(library size: {info['library_size']})")

    # Phase 3: Phase transitions — should form another Niche
    print("\nPhase 3: Phase transitions (steps 61–90)")
    for step in range(61, 91):
        state = generate_neuron_state(step, config.n_neurons, mode="jump")
        info = maia.step(state)
        if info["new_niche_formed"]:
            print(f"  Step {step:3d}: ✓ New Niche formed → "
                  f"[{info['new_niche_id']}] {info['new_niche_label']}  "
                  f"(library size: {info['library_size']})")

    # Phase 4: Mixed dynamics — tests composition of existing Niches
    print("\nPhase 4: Mixed dynamics — testing composition (steps 91–120)")
    composition_count = 0
    for step in range(91, 121):
        state = generate_neuron_state(step, config.n_neurons, mode="mixed")
        info = maia.step(state)
        if info["selection_mode"] == "composition":
            composition_count += 1
        if info["new_niche_formed"]:
            print(f"  Step {step:3d}: ✓ New Niche formed → "
                  f"[{info['new_niche_id']}] {info['new_niche_label']}  "
                  f"(library size: {info['library_size']})")

    print(f"  Composition used in {composition_count}/30 steps")

    # Final summary
    print()
    print(maia.summary())

    # Failure diagnosis demo
    print("Failure Diagnosis Demo")
    print("-" * 50)
    unknown = torch.randn(config.n_neurons) * 10  # very OOD context
    diagnosis = maia.diagnose(unknown)
    print(f"  Failure type: {diagnosis['failure']}")
    print(f"  Detail:       {diagnosis['detail']}")

    # Hierarchy
    if maia.library.m > 0:
        print("\nNiche Hierarchy (root Niches):")
        for root in maia.library.get_root_niches():
            print(maia.library.get_hierarchy(root.id))


if __name__ == "__main__":
    run_demo()