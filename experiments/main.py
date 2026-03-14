"""
main.py — MAIA System
======================
Mechanism-Aware Intelligence Architecture

Wires together:
  - CentralAwarenessHub    (hub.py)
  - NicheFormationGate     (formation_gate.py)
  - NicheLibrary           (niche.py)
  - ProvisionalSpace       (provisional.py)

The full loop (v4.0 — three-state epistemic system):

  1. Receive neuron state from environment
  2. Hub computes change vector + sparse correlations
  3. Hub decomposes change into explained + residual
  4. If residual is large → intake into Provisional Space
  5. Provisional Space accumulates evidence across encounters
  6. When promotion conditions met → pass to Gram-Schmidt gate
  7. If gate admits → add to Niche library
  8. Library selects / composes operative Niche for current context
  9. Hub modulates weights based on both signals
  10. Repeat

This is the minimal end-to-end MAIA system.
No task head yet — this is the Foundation Model core.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from hub import CentralAwarenessHub
from formation_gate import NicheFormationGate
from niche import NicheLibrary, Niche
from provisional import ProvisionalSpace
from monotonic_detector import MonotonicDetector, ResidualClassifier


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class MAIAConfig:
    n_neurons:              int   = 64      # neuron count
    max_niches:             int   = 32      # max Niches before compression
    top_k_correlations:     int   = 16      # sparse correlation pairs
    residual_threshold:     float = 0.08    # Hub: min residual to flag candidate
    similarity_threshold:   float = 0.85    # Gate: max cosine sim to existing Niches
    coverage_threshold:     float = 0.05    # Gate: min residual norm after projection
    stability_threshold:    float = 0.25    # Gate: min stability score
    history_length:         int   = 12      # Gate: candidate history window
    graph_link_threshold:   float = 0.6     # Library: similarity for graph links
    composition_top_k:      int   = 3       # Library: max Niches to compose
    # Provisional Space
    prov_min_encounters:    int   = 3       # recurrence condition
    prov_min_stability:     float = 0.6     # directional stability condition
    prov_min_diversity:     float = 0.2     # contextual diversity condition
    prov_match_threshold:   float = 0.75    # similarity to match existing candidate
    prov_max_age:           float = 3600.0  # seconds before forced decay
    prov_max_idle:          float = 600.0   # seconds without encounter before decay
    prov_max_candidates:    int   = 64      # max provisional candidates held
    # Monotonic Detector
    mono_min_encounters:    int   = 3
    mono_min_stability:     float = 0.6
    mono_max_variance:      float = 0.05
    mono_match_threshold:   float = 0.6
    # Residual Classifier
    mono_threshold:         float = 0.02
    noise_threshold:        float = 2.0
    classifier_window:      int   = 8


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
            n_niches=max(1, config.max_niches // 4),
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

        self.provisional = ProvisionalSpace(
            n_neurons=config.n_neurons,
            min_encounters=config.prov_min_encounters,
            min_stability=config.prov_min_stability,
            min_context_diversity=config.prov_min_diversity,
            match_threshold=config.prov_match_threshold,
            max_age=config.prov_max_age,
            max_idle=config.prov_max_idle,
            max_candidates=config.prov_max_candidates,
        )

        self.monotonic = MonotonicDetector(
            n_neurons=config.n_neurons,
            min_encounters=config.mono_min_encounters,
            min_stability=config.mono_min_stability,
            max_norm_variance=config.mono_max_variance,
            match_threshold=config.mono_match_threshold,
        )

        self.classifier = ResidualClassifier(
            monotonic_threshold=config.mono_threshold,
            noise_threshold=config.noise_threshold,
            window=config.classifier_window,
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
        Full MAIA update step (v4.0 — three-state epistemic system).

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

        step_info["change_norm"]       = hub_result["change_vector"].norm().item()
        step_info["explained_norm"]    = hub_result["explained"].norm().item()
        step_info["residual_norm"]     = hub_result["residual"].norm().item()
        step_info["candidate_flagged"] = hub_result["candidate_niche"]

        # ── 2. Residual Classification + Intake ──────────
        if hub_result["candidate_niche"]:
            residual = hub_result["residual"]
            band = self.classifier.classify(residual)
            if band == 'monotonic':
                self.monotonic.intake(residual, neuron_state)
            elif band == 'provisional':
                self.provisional.intake(residual, neuron_state)
            # noise → discard
            step_info["residual_band"] = band

        step_info["provisional_size"] = self.provisional.size
        step_info["monotonic_size"]   = self.monotonic.size

        # ── 3. Both pathways → Gate: check promotions ────
        new_niche_formed = False
        gate_result = None

        ready_candidates = self.provisional.step() + self.monotonic.step()
        step_info["promoted_this_step"] = len(ready_candidates)

        for cand in ready_candidates:
            vec = self.provisional.promote(cand.id)
            if vec is not None:
                gate_result = self.gate.evaluate(vec)
                if gate_result.admitted:
                    new_niche = self.library.add_niche(
                        gate_result.orthogonal_vector,
                        label=f"niche_{self.library.m}",
                    )
                    new_niche_formed = True
                    step_info["new_niche_id"]    = new_niche.id
                    step_info["new_niche_label"] = new_niche.label

        step_info["gate_admitted"]    = gate_result.admitted if gate_result else None
        step_info["gate_reason"]      = gate_result.failure_reason if gate_result else None
        step_info["new_niche_formed"] = new_niche_formed
        step_info["library_size"]     = self.library.m

        # ── 4. Library: select / compose operative Niche ─
        if self.library.m > 0:
            if self.library.m >= self.config.composition_top_k:
                composed, active_niches = self.library.compose(neuron_state)
                step_info["selection_mode"]   = "composition"
                step_info["active_niches"]    = [n.id for n in active_niches]
                step_info["operative_vector"] = composed
            else:
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
        Checks provisional space first — context may be pending promotion.
        """
        # Check if context matches a provisional candidate
        match = self.provisional._find_match(context)
        if match is not None:
            _, diag = self.provisional.check_promotion(match)
            return {
                "failure": "provisional_pending",
                "candidate_id": match.id,
                "encounter_count": match.encounter_count,
                "detail": f"Context matches a provisional candidate "
                          f"(id={match.id}, encounters={match.encounter_count}). "
                          f"Not yet promoted — still accumulating evidence.",
                **diag
            }

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

        hub_summary = self.hub.get_attribution_summary()

        lines = [
            "=" * 50,
            "MAIA System Summary (v4.0)",
            "=" * 50,
            f"  Steps run:              {self.step_count}",
            f"  Active Niches:          {self.library.m}",
            f"  Candidates flagged:     {total_candidates}",
            f"  Niches admitted:        {total_admitted}",
            f"  Candidates rejected:    {total_rejected}",
            f"  Most active Niche:      {hub_summary['most_active_niche']}",
            f"  Least explained neuron: {hub_summary['least_explained_neuron']}",
            "",
            self.provisional.summary(),
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
    """
    t = step * 0.1

    if mode == "linear":
        base = torch.linspace(0, 1, n) * t * 0.1
        return base + torch.randn(n) * 0.01

    elif mode == "oscillatory":
        base = torch.sin(torch.linspace(0, 3.14, n) * t)
        return base + torch.randn(n) * 0.01

    elif mode == "jump":
        phase = (step // 20) % 2
        base = torch.ones(n) * phase
        base[:n//2] *= -1 if phase else 1
        return base + torch.randn(n) * 0.02

    elif mode == "mixed":
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
        stability_threshold=0.15,
        coverage_threshold=0.03,
        history_length=6,
        prov_min_encounters=3,
        prov_min_stability=0.5,
        prov_min_diversity=0.15,
        prov_match_threshold=0.75,
        prov_max_age=3600,
        prov_max_idle=600,
    )

    maia = MAIA(config)

    print("MAIA v4.0 — End-to-End Demonstration")
    print("=" * 50)
    print(f"Neurons: {config.n_neurons}  |  Max Niches: {config.max_niches}")
    print(f"Provisional: min_encounters={config.prov_min_encounters}, "
          f"min_stability={config.prov_min_stability}")
    print()

    phases = [
        ("linear",      range(1, 31),   "Phase 1: Linear dynamics"),
        ("oscillatory", range(31, 61),  "Phase 2: Oscillatory dynamics"),
        ("jump",        range(61, 91),  "Phase 3: Phase transitions"),
        ("mixed",       range(91, 121), "Phase 4: Mixed dynamics"),
    ]

    for mode, steps, label in phases:
        print(f"\n{label}")
        composition_count = 0
        for step in steps:
            state = generate_neuron_state(step, config.n_neurons, mode=mode)
            info = maia.step(state)
            if info["new_niche_formed"]:
                print(f"  Step {step:3d}: ✓ Niche promoted → "
                      f"[{info['new_niche_id']}] {info['new_niche_label']}  "
                      f"(library: {info['library_size']}, "
                      f"provisional: {info['provisional_size']})")
            if info["selection_mode"] == "composition":
                composition_count += 1
        if mode == "mixed":
            print(f"  Composition used: {composition_count}/{len(steps)} steps")

    print()
    print(maia.summary())

    # Failure diagnosis
    print("Failure Diagnosis Demo")
    print("-" * 50)
    unknown = torch.randn(config.n_neurons) * 10
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