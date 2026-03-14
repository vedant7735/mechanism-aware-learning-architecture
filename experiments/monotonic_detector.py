"""
monotonic_detector.py — Monotonic Detector
============================================
Mechanism-Aware Intelligence Architecture (MAIA)

The Monotonic Detector is the second intake pathway in the
Residual Classification Layer (v5.0).

It handles residuals that are too consistent to satisfy the
Provisional Space's contextual diversity condition, but too
persistent to be noise.

The key insight:
  A perfectly consistent residual is its own evidence.
  If the same signal appears in the same direction repeatedly,
  that IS the most reliable mechanism possible.
  It does not need diverse contexts to prove its generality —
  it IS context-independent by nature.

Promotion conditions (simpler than Provisional Space):
  1. Recurrence       — seen a minimum number of times
  2. Directional stability — vector direction consistent across encounters
  3. Low variation confirmed — norm variance stays below monotonic threshold

No diversity condition. Promotion is faster than Provisional Space by design.

The Residual Classification Layer routes residuals here when:
  variation_score < monotonic_threshold

Everything above that threshold goes to Provisional Space.
Everything above noise_threshold is discarded.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
import time


# ─────────────────────────────────────────────
# Monotonic Candidate
# ─────────────────────────────────────────────

@dataclass
class MonotonicCandidate:
    """
    A single entry in the Monotonic Detector.

    Simpler than ProvisionalCandidate — no context history needed.
    Tracks direction and norm variance only.
    """
    id:                 int
    vector:             torch.Tensor        # current best estimate
    created_at:         float
    last_seen:          float
    encounter_count:    int = 1
    direction_history:  list = field(default_factory=list)
    norm_history:       list = field(default_factory=list)   # for variance check

    def __post_init__(self):
        norm = self.vector.norm()
        self.direction_history.append(F.normalize(self.vector, dim=0).detach())
        self.norm_history.append(norm.item())

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    @property
    def time_since_seen(self) -> float:
        return time.time() - self.last_seen

    @property
    def norm_variance(self) -> float:
        if len(self.norm_history) < 2:
            return 0.0
        mean = sum(self.norm_history) / len(self.norm_history)
        variance = sum((x - mean) ** 2 for x in self.norm_history) / len(self.norm_history)
        return variance

    def __repr__(self):
        return (
            f"MonotonicCandidate(id={self.id}, "
            f"encounters={self.encounter_count}, "
            f"norm_variance={self.norm_variance:.4f}, "
            f"age={self.age:.1f}s)"
        )


# ─────────────────────────────────────────────
# Monotonic Detector
# ─────────────────────────────────────────────

class MonotonicDetector:
    def __init__(
        self,
        n_neurons:              int,
        # Promotion thresholds
        min_encounters:         int   = 3,
        min_stability:          float = 0.7,
        max_norm_variance:      float = 0.05,   # confirmed low variation
        # Matching
        match_threshold:        float = 0.6,    # lower than provisional — monotonic signals are similar by nature
        # Decay
        max_age:                float = 300.0,
        max_idle:               float = 60.0,
        # Capacity
        max_candidates:         int   = 32,     # smaller than provisional — monotonic library stays lean
    ):
        """
        Args:
            n_neurons:          Dimension of each candidate vector
            min_encounters:     Minimum re-encounters for recurrence condition
            min_stability:      Minimum mean cosine similarity across direction history
            max_norm_variance:  Maximum allowed norm variance (confirms low variation)
            match_threshold:    Cosine similarity to match incoming to existing candidate
            max_age:            Maximum time a candidate can stay (seconds)
            max_idle:           Maximum idle time before decay (seconds)
            max_candidates:     Maximum candidates held at once
        """
        self.n = n_neurons
        self.min_encounters = min_encounters
        self.min_stability = min_stability
        self.max_norm_variance = max_norm_variance
        self.match_threshold = match_threshold
        self.max_age = max_age
        self.max_idle = max_idle
        self.max_candidates = max_candidates

        self.candidates: dict[int, MonotonicCandidate] = {}
        self._next_id = 0

        self.promoted: list[int] = []
        self.decayed:  list[int] = []

    @property
    def size(self) -> int:
        return len(self.candidates)

    # ─────────────────────────────────────────────
    # Intake
    # ─────────────────────────────────────────────

    def intake(
        self,
        candidate_vector: torch.Tensor,
        context: torch.Tensor,          # accepted for API consistency, not used
    ) -> Optional[MonotonicCandidate]:
        """
        Receive a low-variation residual from the Residual Classification Layer.

        Args:
            candidate_vector: (n,) — residual classified as monotonic
            context:          (n,) — not used, kept for API consistency with ProvisionalSpace

        Returns:
            The candidate that was updated or created
        """
        assert candidate_vector.shape == (self.n,)

        match = self._find_match(candidate_vector)

        if match is not None:
            self._update_candidate(match, candidate_vector)
            return match
        else:
            if self.size >= self.max_candidates:
                self._decay_oldest()
            return self._create_candidate(candidate_vector)

    def _find_match(
        self, vector: torch.Tensor
    ) -> Optional[MonotonicCandidate]:
        """Find an existing candidate similar enough to the incoming vector."""
        if self.size == 0:
            return None

        vector_norm = F.normalize(vector, dim=0)
        best_sim = -1.0
        best_candidate = None

        for cand in self.candidates.values():
            cand_norm = F.normalize(cand.vector, dim=0)
            sim = (vector_norm @ cand_norm).item()
            if sim > best_sim:
                best_sim = sim
                best_candidate = cand

        if best_sim >= self.match_threshold:
            return best_candidate
        return None

    def _create_candidate(
        self, vector: torch.Tensor
    ) -> MonotonicCandidate:
        """Create a new monotonic candidate."""
        cand_id = self._next_id
        self._next_id += 1

        now = time.time()
        cand = MonotonicCandidate(
            id=cand_id,
            vector=F.normalize(vector, dim=0).detach(),
            created_at=now,
            last_seen=now,
        )

        self.candidates[cand_id] = cand
        return cand

    def _update_candidate(
        self,
        candidate: MonotonicCandidate,
        new_vector: torch.Tensor,
    ) -> None:
        """Update an existing candidate on re-encounter."""
        candidate.encounter_count += 1
        candidate.last_seen = time.time()

        # Running average — same as ProvisionalSpace
        alpha = 0.3
        new_norm = F.normalize(new_vector, dim=0).detach()
        candidate.vector = F.normalize(
            (1 - alpha) * candidate.vector + alpha * new_norm, dim=0
        )

        candidate.direction_history.append(new_norm)
        candidate.norm_history.append(new_vector.norm().item())

    # ─────────────────────────────────────────────
    # Promotion Conditions
    # ─────────────────────────────────────────────

    def check_recurrence(self, candidate: MonotonicCandidate) -> bool:
        """Condition 1: Has this been seen enough times?"""
        return candidate.encounter_count >= self.min_encounters

    def check_directional_stability(
        self, candidate: MonotonicCandidate
    ) -> tuple[bool, float]:
        """
        Condition 2: Is the direction consistent across encounters?
        Same logic as ProvisionalSpace — mean pairwise cosine similarity.
        """
        history = candidate.direction_history
        if len(history) < 2:
            return False, 0.0

        sims = []
        for i in range(len(history)):
            for j in range(i + 1, len(history)):
                sim = (history[i] @ history[j]).item()
                sims.append(sim)

        stability = sum(sims) / len(sims) if sims else 0.0
        return stability >= self.min_stability, stability

    def check_low_variation(
        self, candidate: MonotonicCandidate
    ) -> tuple[bool, float]:
        """
        Condition 3: Confirmed low variation.
        Norm variance must stay below the monotonic threshold.
        This is what qualifies it for this pathway over provisional space.
        """
        variance = candidate.norm_variance
        return variance <= self.max_norm_variance, variance

    def check_promotion(
        self, candidate: MonotonicCandidate
    ) -> tuple[bool, dict]:
        """
        Run all three promotion conditions.
        No diversity condition — that's the whole point of this pathway.
        """
        rec_ok = self.check_recurrence(candidate)
        stab_ok, stab_score = self.check_directional_stability(candidate)
        var_ok, variance = self.check_low_variation(candidate)

        ready = rec_ok and stab_ok and var_ok

        return ready, {
            "recurrence_ok":    rec_ok,
            "encounter_count":  candidate.encounter_count,
            "stability_ok":     stab_ok,
            "stability_score":  stab_score,
            "low_variation_ok": var_ok,
            "norm_variance":    variance,
        }

    # ─────────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────────

    def step(self) -> list[MonotonicCandidate]:
        """
        Run one update step.
        Decay expired candidates, return promotion-ready ones.
        """
        self._decay_expired()

        ready = []
        for cand in list(self.candidates.values()):
            promote, _ = self.check_promotion(cand)
            if promote:
                ready.append(cand)

        return ready

    def promote(self, candidate_id: int) -> Optional[torch.Tensor]:
        """Remove candidate and return its vector for the Gram-Schmidt gate."""
        if candidate_id not in self.candidates:
            return None

        cand = self.candidates.pop(candidate_id)
        self.promoted.append(candidate_id)
        return cand.vector

    # ─────────────────────────────────────────────
    # Decay
    # ─────────────────────────────────────────────

    def _decay_expired(self) -> None:
        to_remove = []
        for cid, cand in self.candidates.items():
            if cand.age > self.max_age:
                to_remove.append(cid)
            elif cand.time_since_seen > self.max_idle:
                to_remove.append(cid)

        for cid in to_remove:
            self.candidates.pop(cid)
            self.decayed.append(cid)

    def _decay_oldest(self) -> None:
        if not self.candidates:
            return
        oldest_id = min(self.candidates, key=lambda k: self.candidates[k].created_at)
        self.candidates.pop(oldest_id)
        self.decayed.append(oldest_id)

    # ─────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"MonotonicDetector Summary",
            f"  Active candidates:  {self.size}",
            f"  Total promoted:     {len(self.promoted)}",
            f"  Total decayed:      {len(self.decayed)}",
        ]
        if self.candidates:
            most_seen = max(self.candidates.values(), key=lambda c: c.encounter_count)
            lines.append(
                f"  Most encountered:   id={most_seen.id} "
                f"({most_seen.encounter_count} times, "
                f"variance={most_seen.norm_variance:.4f})"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Residual Classification Layer
# ─────────────────────────────────────────────

class ResidualClassifier:
    """
    Band-stop filter that routes incoming residuals to the correct pathway.

    Three bands:
      Low variation  (norm_variance < monotonic_threshold) → MonotonicDetector
      Mid variation  (monotonic_threshold ≤ variance < noise_threshold) → ProvisionalSpace
      High variation (variance ≥ noise_threshold) → Discard

    Variation is measured as the rolling variance of residual norms
    across recent steps.
    """

    def __init__(
        self,
        monotonic_threshold: float = 0.02,
        noise_threshold:     float = 2.0,
        window:              int   = 8,
    ):
        self.monotonic_threshold = monotonic_threshold
        self.noise_threshold = noise_threshold
        self.window = window
        self._norm_history: list[float] = []

    def classify(self, residual: torch.Tensor) -> str:
        """
        Classify a residual vector into one of three bands.

        Returns:
            'monotonic'   — route to MonotonicDetector
            'provisional' — route to ProvisionalSpace
            'noise'       — discard
        """
        norm = residual.norm().item()
        self._norm_history.append(norm)
        if len(self._norm_history) > self.window:
            self._norm_history.pop(0)

        if len(self._norm_history) < 2:
            # Not enough history — default to provisional
            return 'provisional'

        mean = sum(self._norm_history) / len(self._norm_history)
        variance = sum((x - mean) ** 2 for x in self._norm_history) / len(self._norm_history)

        if variance < self.monotonic_threshold:
            return 'monotonic'
        elif variance >= self.noise_threshold:
            return 'noise'
        else:
            return 'provisional'


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    detector = MonotonicDetector(
        n_neurons=32,
        min_encounters=3,
        min_stability=0.6,
        max_norm_variance=0.1,
        match_threshold=0.6,
    )
    classifier = ResidualClassifier(
        monotonic_threshold=0.02,
        noise_threshold=2.0,
    )

    print("MonotonicDetector + ResidualClassifier — Demonstration")
    print("=" * 50)

    # Simulate a consistent linear residual — should promote
    base = torch.zeros(32)
    base[:8] = 1.0  # consistent direction
    print("\nSimulating consistent linear residual (should promote):")

    for i in range(6):
        residual = base + torch.randn(32) * 0.01  # tiny noise
        context  = torch.randn(32)
        band     = classifier.classify(residual)
        print(f"  Step {i+1}: band={band}", end="")

        if band == 'monotonic':
            cand = detector.intake(residual, context)
            ready, diag = detector.check_promotion(cand)
            print(f" | id={cand.id} enc={cand.encounter_count} "
                  f"stab={diag['stability_score']:.3f} "
                  f"var={diag['norm_variance']:.4f} ready={ready}")
        else:
            print()

    print()
    ready_candidates = detector.step()
    print(f"Ready for promotion: {[c.id for c in ready_candidates]}")

    for cand in ready_candidates:
        vec = detector.promote(cand.id)
        print(f"Promoted id={cand.id} → vector norm={vec.norm():.4f}")

    # Simulate noisy residual — should go to provisional or discard
    print("\nSimulating noisy residual (should not go to monotonic):")
    for i in range(4):
        residual = torch.randn(32) * (i + 1)  # growing noise
        band = classifier.classify(residual)
        print(f"  Step {i+1}: band={band}")

    print(f"\n{detector.summary()}")