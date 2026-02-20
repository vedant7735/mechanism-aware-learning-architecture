"""
formation_gate.py — Niche Formation Gate
==========================================
Mechanism-Aware Intelligence Architecture (MAIA)

This module decides whether a candidate vector (the residual from the Hub)
qualifies as a new Niche in the mechanism library.

The gate operates in two steps:

  Step 1 — Gram-Schmidt Orthogonalization:
    Project the candidate against all existing Niche columns.
    Subtract out what existing Niches already explain.
    Only the genuinely new residual is evaluated.

  Step 2 — Four Invariance Conditions (all must pass):
    1. Generalization  — not specific to one observation
    2. Stability       — consistent across recent history
    3. Coverage        — residual magnitude is meaningful
    4. Decorrelation   — sufficiently orthogonal to existing Niches

If all four pass → candidate is admitted as a new Niche column.
If not → candidate is merged, compressed, or discarded.

The existing Niche columns are not passive — they actively define
what counts as "new" by determining what is already explained.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class FormationResult:
    """Result returned by the formation gate."""
    admitted:           bool            # Whether the candidate became a Niche
    orthogonal_vector:  torch.Tensor    # Candidate after Gram-Schmidt projection
    residual_norm:      float           # Magnitude of the orthogonal residual
    max_similarity:     float           # Cosine similarity to closest existing Niche
    stability_score:    float           # Consistency score across history
    generalization_ok:  bool            # Passed generalization check
    stability_ok:       bool            # Passed stability check
    coverage_ok:        bool            # Passed coverage check
    decorrelation_ok:   bool            # Passed decorrelation check
    failure_reason:     Optional[str]   # Why it was rejected (if rejected)


class NicheFormationGate:
    def __init__(
        self,
        n_neurons: int,
        max_niches: int = 64,
        similarity_threshold: float = 0.85,
        coverage_threshold: float = 0.05,
        stability_threshold: float = 0.6,
        history_length: int = 16,
    ):
        """
        Args:
            n_neurons:              Dimension of each Niche vector (n)
            max_niches:             Maximum allowed Niches before compression kicks in
            similarity_threshold:   Max cosine similarity to existing Niches (decorrelation gate)
            coverage_threshold:     Min residual norm after projection (coverage gate)
            stability_threshold:    Min stability score across history (stability gate)
            history_length:         How many past candidates to track for stability check
        """
        self.n = n_neurons
        self.max_niches = max_niches
        self.sim_threshold = similarity_threshold
        self.cov_threshold = coverage_threshold
        self.stab_threshold = stability_threshold
        self.history_length = history_length

        # The mechanism library — columns are Niche vectors
        # Starts empty; grows as Niches are admitted
        self.niche_library: torch.Tensor = torch.zeros(n_neurons, 0)  # (n, 0) initially

        # Candidate history for stability checking
        # Stores recent candidate vectors to measure consistency
        self.candidate_history: list[torch.Tensor] = []

        # Formation log — useful for debugging and paper experiments
        self.formation_log: list[FormationResult] = []

    @property
    def m(self) -> int:
        """Current number of Niches in the library."""
        return self.niche_library.shape[1]

    # ─────────────────────────────────────────────
    # Step 1: Gram-Schmidt Orthogonalization
    # ─────────────────────────────────────────────

    def gram_schmidt_project(self, candidate: torch.Tensor) -> torch.Tensor:
        """
        Project the candidate vector against all existing Niche columns
        and subtract out their contributions.

        What remains is the component of the candidate that is genuinely
        new — not explained by any existing mechanism.

        This is classical Gram-Schmidt orthogonalization applied to the
        mechanism library.

        Args:
            candidate: (n,) — raw candidate vector (residual from Hub)

        Returns:
            orthogonal: (n,) — candidate with existing Niche components removed
        """
        orthogonal = candidate.clone()

        if self.m == 0:
            # No existing Niches — candidate is fully new by definition
            return orthogonal

        for i in range(self.m):
            niche_col = self.niche_library[:, i]  # (n,)
            niche_col_norm = niche_col.norm()

            if niche_col_norm < 1e-8:
                continue  # Skip degenerate Niche columns

            # Project out the component along this Niche
            projection = (orthogonal @ niche_col) / (niche_col_norm ** 2)
            orthogonal = orthogonal - projection * niche_col

        return orthogonal

    # ─────────────────────────────────────────────
    # Step 2: Four Invariance Conditions
    # ─────────────────────────────────────────────

    def check_generalization(self, candidate: torch.Tensor) -> bool:
        """
        Condition 1: Generalization
        The candidate should not be a spike or degenerate vector —
        it should distribute its signal across multiple dimensions,
        suggesting it captures a broad dynamic rather than a single observation.

        Measured by: ratio of max component to mean component.
        A healthy mechanism vector is not dominated by one entry.
        """
        abs_vals = candidate.abs()
        if abs_vals.sum() < 1e-8:
            return False

        max_component = abs_vals.max().item()
        mean_component = abs_vals.mean().item()

        # If one component dominates massively, it's too specific
        dominance_ratio = max_component / (mean_component + 1e-8)
        return dominance_ratio < (self.n * 0.5)  # generous threshold

    def check_stability(self, candidate: torch.Tensor) -> float:
        """
        Condition 2: Stability
        The candidate should be consistent with recent candidate history.
        A stable mechanism appears repeatedly with similar direction —
        it is not a one-off noise residual.

        Returns a stability score in [0, 1].
        Higher = more consistent with history.
        """
        # Update history
        self.candidate_history.append(candidate.detach())
        if len(self.candidate_history) > self.history_length:
            self.candidate_history.pop(0)

        if len(self.candidate_history) < 2:
            # Not enough history yet — give benefit of the doubt
            return 1.0

        # Measure average cosine similarity of current candidate to history
        candidate_norm = candidate.norm()
        if candidate_norm < 1e-8:
            return 0.0

        similarities = []
        for past in self.candidate_history[:-1]:  # exclude current
            past_norm = past.norm()
            if past_norm < 1e-8:
                continue
            sim = (candidate @ past) / (candidate_norm * past_norm)
            similarities.append(sim.item())

        if not similarities:
            return 1.0

        return max(0.0, sum(similarities) / len(similarities))

    def check_coverage(self, orthogonal: torch.Tensor) -> bool:
        """
        Condition 3: Coverage
        After Gram-Schmidt projection, the remaining vector must be
        large enough to be meaningful — not just floating point noise.

        A small residual means existing Niches already explain this dynamic.
        """
        return orthogonal.norm().item() > self.cov_threshold

    def check_decorrelation(self, orthogonal: torch.Tensor) -> tuple[bool, float]:
        """
        Condition 4: Decorrelation
        The orthogonal candidate must have low cosine similarity to all
        existing Niche columns.

        Even after Gram-Schmidt, we verify — the projection removes linear
        components but similarity in direction still matters.

        Returns:
            (passes: bool, max_similarity: float)
        """
        if self.m == 0:
            return True, 0.0

        orthogonal_norm = orthogonal.norm()
        if orthogonal_norm < 1e-8:
            return False, 1.0

        max_sim = 0.0
        for i in range(self.m):
            niche_col = self.niche_library[:, i]
            niche_norm = niche_col.norm()
            if niche_norm < 1e-8:
                continue
            sim = abs((orthogonal @ niche_col) / (orthogonal_norm * niche_norm)).item()
            max_sim = max(max_sim, sim)

        return max_sim < self.sim_threshold, max_sim

    # ─────────────────────────────────────────────
    # Gate: Full Evaluation
    # ─────────────────────────────────────────────

    def evaluate(self, candidate: torch.Tensor) -> FormationResult:
        """
        Run the full formation gate on a candidate vector.

        Args:
            candidate: (n,) — residual from the Hub flagged as a Niche candidate

        Returns:
            FormationResult with full diagnostic information
        """
        assert candidate.shape == (self.n,), (
            f"Expected candidate of shape ({self.n},), got {candidate.shape}"
        )

        # Step 1: Gram-Schmidt projection
        orthogonal = self.gram_schmidt_project(candidate)
        residual_norm = orthogonal.norm().item()

        # Step 2: Four conditions
        gen_ok  = self.check_generalization(orthogonal)
        stab    = self.check_stability(orthogonal)
        stab_ok = stab >= self.stab_threshold
        cov_ok  = self.check_coverage(orthogonal)
        dec_ok, max_sim = self.check_decorrelation(orthogonal)

        admitted = gen_ok and stab_ok and cov_ok and dec_ok

        # Determine failure reason if rejected
        failure_reason = None
        if not admitted:
            reasons = []
            if not gen_ok:  reasons.append("failed generalization (too localized)")
            if not stab_ok: reasons.append(f"failed stability (score={stab:.3f})")
            if not cov_ok:  reasons.append(f"failed coverage (norm={residual_norm:.4f})")
            if not dec_ok:  reasons.append(f"failed decorrelation (max_sim={max_sim:.3f})")
            failure_reason = "; ".join(reasons)

        result = FormationResult(
            admitted=admitted,
            orthogonal_vector=orthogonal,
            residual_norm=residual_norm,
            max_similarity=max_sim,
            stability_score=stab,
            generalization_ok=gen_ok,
            stability_ok=stab_ok,
            coverage_ok=cov_ok,
            decorrelation_ok=dec_ok,
            failure_reason=failure_reason,
        )

        # If admitted, add to library
        if admitted:
            self._admit_niche(orthogonal)

        self.formation_log.append(result)
        return result

    # ─────────────────────────────────────────────
    # Library Management
    # ─────────────────────────────────────────────

    def _admit_niche(self, vector: torch.Tensor) -> None:
        """
        Normalize and append the new Niche column to the library.
        """
        normalized = F.normalize(vector, dim=0)
        self.niche_library = torch.cat(
            [self.niche_library, normalized.unsqueeze(1)], dim=1
        )

        # If library is getting large, trigger compression
        if self.m > self.max_niches:
            self._compress_library()

    def _compress_library(self) -> None:
        """
        When the library exceeds max_niches, merge the most similar pair.
        This keeps the library sparse and prevents Niche explosion.
        """
        if self.m < 2:
            return

        # Find the most similar pair of Niches
        lib = self.niche_library  # (n, m)
        gram = lib.T @ lib        # (m, m) — dot products between all Niche pairs

        # Zero diagonal (self-similarity)
        gram.fill_diagonal_(-1.0)

        # Find most similar pair
        idx = gram.argmax()
        i, j = idx // self.m, idx % self.m
        i, j = i.item(), j.item()

        if i == j:
            return

        # Merge: average and renormalize
        merged = F.normalize(lib[:, i] + lib[:, j], dim=0)

        # Remove both and add merged
        cols = [c for c in range(self.m) if c != i and c != j]
        kept = lib[:, cols]
        self.niche_library = torch.cat([kept, merged.unsqueeze(1)], dim=1)

    def get_library_summary(self) -> dict:
        """
        Returns diagnostic info about the current Niche library.
        """
        if self.m == 0:
            return {"n_niches": 0, "admitted": 0, "rejected": 0}

        lib = self.niche_library
        gram = lib.T @ lib
        gram.fill_diagonal_(0)

        admitted = sum(1 for r in self.formation_log if r.admitted)
        rejected = sum(1 for r in self.formation_log if not r.admitted)

        return {
            "n_niches": self.m,
            "admitted": admitted,
            "rejected": rejected,
            "max_inter_niche_similarity": gram.abs().max().item(),
            "mean_inter_niche_similarity": gram.abs().mean().item(),
        }


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    gate = NicheFormationGate(
        n_neurons=32,
        max_niches=16,
        similarity_threshold=0.85,
        coverage_threshold=0.05,
        stability_threshold=0.3,  # low for demo — more admissions
        history_length=8,
    )

    print("Niche Formation Gate — Demonstration")
    print("=" * 45)

    admitted_count = 0
    for trial in range(20):
        # Simulate candidates: mix of structured and noisy residuals
        if trial % 4 == 0:
            # Structured candidate — likely to pass
            candidate = torch.zeros(32)
            candidate[trial % 8: trial % 8 + 4] = 1.0
            candidate += torch.randn(32) * 0.1
        else:
            # Noisy candidate — likely to fail coverage or decorrelation
            candidate = torch.randn(32) * 0.03

        result = gate.evaluate(candidate)

        status = "✓ ADMITTED" if result.admitted else f"✗ REJECTED"
        reason = f" — {result.failure_reason}" if result.failure_reason else ""
        print(f"Trial {trial+1:2d}: {status}{reason}")
        print(f"         norm={result.residual_norm:.4f}  "
              f"sim={result.max_similarity:.3f}  "
              f"stab={result.stability_score:.3f}")

        if result.admitted:
            admitted_count += 1

    print()
    summary = gate.get_library_summary()
    print("Library Summary:")
    print(f"  Active Niches:              {summary['n_niches']}")
    print(f"  Total admitted:             {summary['admitted']}")
    print(f"  Total rejected:             {summary['rejected']}")
    print(f"  Max inter-Niche similarity: {summary['max_inter_niche_similarity']:.4f}")
    print(f"  Mean inter-Niche similarity:{summary['mean_inter_niche_similarity']:.4f}")