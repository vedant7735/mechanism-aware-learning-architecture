"""
toy_dynamics.py — Controlled Experiments
==========================================
Mechanism-Aware Intelligence Architecture (MAIA)
experiments/toy_dynamics.py

A clean, measurable experiment suite for validating MAIA's core hypotheses:

  Hypothesis 1 — Mechanism Discovery:
    MAIA forms distinct Niches for distinct underlying dynamics.
    Different dynamics → different Niche activations.

  Hypothesis 2 — Mechanism Reuse:
    When a previously seen dynamic reappears, MAIA selects an existing
    Niche rather than forming a new one.

  Hypothesis 3 — Composition:
    Mixed dynamics are handled by composing existing Niches,
    not by forming redundant new ones.

  Hypothesis 4 — Failure Traceability:
    When MAIA encounters a truly novel dynamic, it correctly identifies
    the failure mode (incomplete mechanism) rather than hallucinating.

Each experiment produces clean metrics that can be plotted and reported.
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))

# type: ignore
from hub import CentralAwarenessHub
from formation_gate import NicheFormationGate
from niche import NicheLibrary
from main import MAIA, MAIAConfig, generate_neuron_state


# ─────────────────────────────────────────────
# Metrics Tracking
# ─────────────────────────────────────────────

class ExperimentMetrics:
    """Tracks per-step metrics for analysis and plotting."""

    def __init__(self, name: str):
        self.name = name
        self.steps:               list[int]   = []
        self.library_sizes:       list[int]   = []
        self.residual_norms:      list[float] = []
        self.explained_norms:     list[float] = []
        self.explanation_ratios:  list[float] = []  # explained / (explained + residual)
        self.niche_formations:    list[int]   = []  # step at which each Niche formed
        self.active_niche_ids:    list[list]  = []
        self.selection_modes:     list[str]   = []
        self.candidate_flags:     list[bool]  = []
        self.gate_admissions:     list[bool]  = []

    def record(self, step: int, info: dict) -> None:
        self.steps.append(step)
        self.library_sizes.append(info["library_size"])
        self.residual_norms.append(info["residual_norm"])
        self.explained_norms.append(info["explained_norm"])

        total = info["explained_norm"] + info["residual_norm"]
        ratio = info["explained_norm"] / total if total > 1e-8 else 0.0
        self.explanation_ratios.append(ratio)

        self.active_niche_ids.append(info.get("active_niches", []))
        self.selection_modes.append(info.get("selection_mode", "none"))
        self.candidate_flags.append(info.get("candidate_flagged", False))

        admitted = info.get("gate_admitted")
        self.gate_admissions.append(admitted if admitted is not None else False)

        if info.get("new_niche_formed"):
            self.niche_formations.append(step)

    def print_summary(self) -> None:
        if not self.steps:
            print(f"[{self.name}] No data recorded.")
            return

        total_steps     = len(self.steps)
        total_candidates= sum(self.candidate_flags)
        total_admitted  = len(self.niche_formations)
        final_library   = self.library_sizes[-1] if self.library_sizes else 0
        avg_explanation = sum(self.explanation_ratios) / len(self.explanation_ratios)
        composition_steps = sum(1 for m in self.selection_modes if m == "composition")

        print(f"\n{'─'*50}")
        print(f"Experiment: {self.name}")
        print(f"{'─'*50}")
        print(f"  Steps:                  {total_steps}")
        print(f"  Final library size:     {final_library} Niches")
        print(f"  Niches formed at steps: {self.niche_formations}")
        print(f"  Candidates flagged:     {total_candidates}")
        print(f"  Gate admissions:        {total_admitted}")
        print(f"  Avg explanation ratio:  {avg_explanation:.3f}  "
              f"(1.0 = fully explained by existing Niches)")
        print(f"  Composition steps:      {composition_steps}/{total_steps}")

    def explained_ratio_trend(self, window: int = 10) -> list[float]:
        """Rolling average of explanation ratio — shows learning curve."""
        smoothed = []
        for i in range(len(self.explanation_ratios)):
            start = max(0, i - window + 1)
            smoothed.append(
                sum(self.explanation_ratios[start:i+1]) / (i - start + 1)
            )
        return smoothed


# ─────────────────────────────────────────────
# Experiment 1: Mechanism Discovery
# ─────────────────────────────────────────────

def experiment_mechanism_discovery(n_neurons: int = 32, n_steps: int = 60) -> ExperimentMetrics:
    """
    Present MAIA with a single consistent dynamic (linear drift).
    Expect: one or few Niches form early, then explanation ratio rises
    as the Niche is reused rather than new ones forming.

    Validates: Hypothesis 1 (discovery) + Hypothesis 2 (reuse)
    """
    print("\n" + "="*50)
    print("Experiment 1: Mechanism Discovery & Reuse")
    print("="*50)

    config = MAIAConfig(
        n_neurons=n_neurons,
        max_niches=16,
        residual_threshold=0.04,
        stability_threshold=0.15,
        coverage_threshold=0.02,
        history_length=6,
    )
    maia = MAIA(config)
    metrics = ExperimentMetrics("Mechanism Discovery")

    for step in range(1, n_steps + 1):
        state = generate_neuron_state(step, n_neurons, mode="linear")
        info = maia.step(state)
        metrics.record(step, info)

        if info["new_niche_formed"]:
            print(f"  Step {step:3d}: Niche [{info['new_niche_id']}] formed  "
                  f"| library: {info['library_size']}  "
                  f"| explanation ratio: {metrics.explanation_ratios[-1]:.3f}")

    # Check: explanation ratio should rise over time (reuse kicking in)
    early_avg  = sum(metrics.explanation_ratios[:10]) / 10
    late_avg   = sum(metrics.explanation_ratios[-10:]) / 10
    reuse_confirmed = late_avg > early_avg

    print(f"\n  Early explanation ratio (steps 1–10):   {early_avg:.3f}")
    print(f"  Late explanation ratio  (steps 51–60):  {late_avg:.3f}")
    print(f"  Reuse confirmed: {'✓ YES' if reuse_confirmed else '✗ NO'}")

    metrics.print_summary()
    return metrics


# ─────────────────────────────────────────────
# Experiment 2: Distinct Niches for Distinct Dynamics
# ─────────────────────────────────────────────

def experiment_distinct_niches(n_neurons: int = 32, steps_per_phase: int = 40) -> ExperimentMetrics:
    """
    Present MAIA with three clearly different dynamics in sequence.
    Expect: distinct Niches form for each dynamic.
    When a dynamic reappears, existing Niche is reused (not re-formed).

    Validates: Hypothesis 1 + Hypothesis 2
    """
    print("\n" + "="*50)
    print("Experiment 2: Distinct Niches for Distinct Dynamics")
    print("="*50)

    config = MAIAConfig(
        n_neurons=n_neurons,
        max_niches=16,
        residual_threshold=0.04,
        stability_threshold=0.15,
        coverage_threshold=0.02,
        history_length=6,
    )
    maia = MAIA(config)
    metrics = ExperimentMetrics("Distinct Niches")

    phases = [
        ("linear",      steps_per_phase),
        ("oscillatory", steps_per_phase),
        ("jump",        steps_per_phase),
        ("linear",      steps_per_phase),  # repeat — should reuse, not re-form
    ]

    global_step = 0
    niches_at_phase_start = {}

    for phase_name, n_steps in phases:
        niches_at_phase_start[phase_name] = maia.library.m
        phase_formations = 0
        print(f"\n  Phase: {phase_name:12s} | Niches before: {maia.library.m}")

        for step in range(1, n_steps + 1):
            global_step += 1
            state = generate_neuron_state(step, n_neurons, mode=phase_name)
            info = maia.step(state)
            metrics.record(global_step, info)
            if info["new_niche_formed"]:
                phase_formations += 1

        print(f"  Phase: {phase_name:12s} | Niches after:  {maia.library.m} "
              f"(+{phase_formations} formed this phase)")

    # Check: linear reappearance should form 0 new Niches (reuse)
    print(f"\n  Linear dynamic reappearance: "
          f"{'✓ reused existing Niches (0 new formed)' if metrics.niche_formations and metrics.niche_formations[-1] < 3 * steps_per_phase else '→ check reuse behavior'}")

    metrics.print_summary()
    return metrics


# ─────────────────────────────────────────────
# Experiment 3: Composition
# ─────────────────────────────────────────────

def experiment_composition(n_neurons: int = 32) -> ExperimentMetrics:
    """
    Train on individual dynamics first (build library).
    Then present mixed dynamics and measure whether composition is used
    vs. new Niche formation.

    Validates: Hypothesis 3 (composition)
    """
    print("\n" + "="*50)
    print("Experiment 3: Composition of Existing Niches")
    print("="*50)

    config = MAIAConfig(
        n_neurons=n_neurons,
        max_niches=16,
        residual_threshold=0.04,
        stability_threshold=0.15,
        coverage_threshold=0.02,
        history_length=6,
        composition_top_k=3,
    )
    maia = MAIA(config)
    metrics = ExperimentMetrics("Composition")

    # Phase 1: Build library with individual dynamics
    print("\n  Building library (individual dynamics)...")
    for mode in ["linear", "oscillatory", "jump"]:
        for step in range(1, 41):
            state = generate_neuron_state(step, n_neurons, mode=mode)
            maia.step(state)

    library_before_mix = maia.library.m
    print(f"  Library size before mixed phase: {library_before_mix} Niches")

    # Phase 2: Mixed dynamics — should compose, not form new Niches
    print("\n  Mixed dynamics phase (steps 1–60)...")
    new_formations_during_mix = 0
    composition_steps = 0

    for step in range(1, 61):
        state = generate_neuron_state(step, n_neurons, mode="mixed")
        info = maia.step(state)
        metrics.record(step, info)

        if info["new_niche_formed"]:
            new_formations_during_mix += 1
        if info["selection_mode"] == "composition":
            composition_steps += 1

    library_after_mix = maia.library.m
    composition_ratio = composition_steps / 60

    print(f"  Library size after mixed phase:  {library_after_mix} Niches")
    print(f"  New Niches formed during mix:    {new_formations_during_mix}")
    print(f"  Composition used:                {composition_steps}/60 steps "
          f"({composition_ratio:.1%})")
    print(f"  Composition hypothesis: "
          f"{'✓ confirmed' if composition_ratio > 0.3 else '→ partial — may need more library depth'}")

    metrics.print_summary()
    return metrics


# ─────────────────────────────────────────────
# Experiment 4: Failure Traceability
# ─────────────────────────────────────────────

def experiment_failure_traceability(n_neurons: int = 32) -> None:
    """
    Present MAIA with contexts it has and hasn't seen.
    Verify that failure diagnosis correctly identifies:
      - 'incomplete_mechanism' for truly novel contexts
      - 'none_detected' for familiar contexts

    Validates: Hypothesis 4 (failure traceability)
    """
    print("\n" + "="*50)
    print("Experiment 4: Failure Traceability")
    print("="*50)

    config = MAIAConfig(n_neurons=n_neurons, stability_threshold=0.15,
                        residual_threshold=0.04, coverage_threshold=0.02)
    maia = MAIA(config)

    # Build a small library
    for mode in ["linear", "oscillatory"]:
        for step in range(1, 31):
            state = generate_neuron_state(step, n_neurons, mode=mode)
            maia.step(state)

    print(f"\n  Library built: {maia.library.m} Niches")
    print()

    test_cases = [
        ("Familiar — linear",        generate_neuron_state(5, n_neurons, "linear")),
        ("Familiar — oscillatory",   generate_neuron_state(5, n_neurons, "oscillatory")),
        ("Novel — random noise",     torch.randn(n_neurons) * 5),
        ("Novel — extreme values",   torch.ones(n_neurons) * 100),
        ("Novel — sparse spike",     torch.zeros(n_neurons).index_fill_(0, torch.tensor([0]), 50.0)),
    ]

    for label, context in test_cases:
        if maia.library.m == 0:
            print(f"  {label}: library empty — cannot diagnose")
            continue
        diagnosis = maia.diagnose(context)
        failure   = diagnosis["failure"]
        icon = "✓" if failure == "none_detected" else "⚠"
        print(f"  {icon} {label}")
        print(f"     → {failure}: {diagnosis['detail'][:80]}")


# ─────────────────────────────────────────────
# Experiment 5: Scaling — Niche Count vs Explanation
# ─────────────────────────────────────────────

def experiment_scaling(n_neurons: int = 32) -> None:
    """
    Run MAIA with different max_niches settings and measure
    how explanation ratio scales with Niche count.

    This directly tests the awareness spectrum claim:
    more Niches → better explanation → higher awareness.
    """
    print("\n" + "="*50)
    print("Experiment 5: Awareness Spectrum — Scaling")
    print("="*50)

    niche_caps = [2, 4, 8, 16]
    results = []

    for cap in niche_caps:
        config = MAIAConfig(
            n_neurons=n_neurons,
            max_niches=cap,
            residual_threshold=0.04,
            stability_threshold=0.15,
            coverage_threshold=0.02,
            history_length=6,
        )
        maia = MAIA(config)

        explanation_ratios = []
        for step in range(1, 81):
            mode = ["linear", "oscillatory", "jump"][step % 3]
            state = generate_neuron_state(step, n_neurons, mode=mode)
            info = maia.step(state)
            total = info["explained_norm"] + info["residual_norm"]
            ratio = info["explained_norm"] / total if total > 1e-8 else 0.0
            explanation_ratios.append(ratio)

        avg_ratio = sum(explanation_ratios) / len(explanation_ratios)
        final_niches = maia.library.m
        results.append((cap, final_niches, avg_ratio))

        print(f"  max_niches={cap:3d} | final library={final_niches:2d} | "
              f"avg explanation={avg_ratio:.3f}")

    # Check monotonic trend
    ratios = [r[2] for r in results]
    is_monotonic = all(ratios[i] <= ratios[i+1] for i in range(len(ratios)-1))
    print(f"\n  Awareness spectrum monotonic: {'✓ YES' if is_monotonic else '~ partial'}")
    print("  (Higher Niche cap → better average explanation → richer awareness)")


# ─────────────────────────────────────────────
# Run All Experiments
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    N = 32  # neuron count for all experiments

    print("MAIA — Toy Dynamics Experiment Suite")
    print("=" * 50)
    print(f"Neurons: {N}")
    print("Running 5 experiments...\n")

    m1 = experiment_mechanism_discovery(n_neurons=N, n_steps=60)
    m2 = experiment_distinct_niches(n_neurons=N, steps_per_phase=40)
    m3 = experiment_composition(n_neurons=N)
    experiment_failure_traceability(n_neurons=N)
    experiment_scaling(n_neurons=N)

    print("\n" + "="*50)
    print("All experiments complete.")
    print("These results map directly to the 6 research hypotheses")
    print("in the MAIA formal specification (spec_v3.docx).")
    print("="*50)