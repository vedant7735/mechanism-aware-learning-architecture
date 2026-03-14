"""
provisional.py — Provisional Space
=====================================
Mechanism-Aware Intelligence Architecture (MAIA)

The Provisional Space is the middle layer of the three-state epistemic system:

  State 1 — Noise:        discarded immediately (below noise threshold)
  State 2 — Provisional:  real signal, accumulating evidence (this file)
  State 3 — Understood:   full Niche, admitted to permanent library

A candidate enters the Provisional Space when it passes the noise threshold
but fails one or more of the four full Niche formation conditions.
It is not noise — it has real signal. But it is not yet a Niche.

Promotion from Provisional → Full Niche requires all three conditions:
  1. Recurrence       — seen a minimum number of times
  2. Directional stability — vector direction consistent across encounters
  3. Contextual diversity  — appeared in sufficiently different contexts

Decay removes candidates that fail to accumulate evidence within a time window.

This makes MAIA significantly less brittle to data ordering and more
faithful to how biological memory consolidation actually works.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
import time


# ─────────────────────────────────────────────
# Provisional Candidate
# ─────────────────────────────────────────────

@dataclass
class ProvisionalCandidate:
    """
    A single entry in the Provisional Space.

    Tracks all evidence needed to decide whether this candidate
    deserves promotion to a full Niche or should be decayed.
    """
    id:                 int
    vector:             torch.Tensor        # current best estimate (updated on re-encounter)
    created_at:         float               # when first seen
    last_seen:          float               # when last encountered
    encounter_count:    int = 1             # how many times seen
    context_history:    list = field(default_factory=list)    # (n,) contexts seen in
    direction_history:  list = field(default_factory=list)    # unit vectors per encounter

    def __post_init__(self):
        # Store first direction
        norm = self.vector.norm()
        if norm > 1e-8:
            self.direction_history.append(F.normalize(self.vector, dim=0).detach())

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    @property
    def time_since_seen(self) -> float:
        return time.time() - self.last_seen

    def __repr__(self):
        return (
            f"ProvisionalCandidate(id={self.id}, "
            f"encounters={self.encounter_count}, "
            f"contexts={len(self.context_history)}, "
            f"age={self.age:.1f}s)"
        )


# ─────────────────────────────────────────────
# Provisional Space
# ─────────────────────────────────────────────

class ProvisionalSpace:
    def __init__(
        self,
        n_neurons:              int,
        # Promotion thresholds
        min_encounters:         int   = 3,      # recurrence condition
        min_stability:          float = 0.7,    # directional stability condition
        min_context_diversity:  float = 0.3,    # contextual diversity condition
        # Similarity threshold for matching incoming candidates to existing ones
        match_threshold:        float = 0.8,
        # Decay settings
        max_age:                float = 300.0,  # seconds before forced decay
        max_idle:               float = 60.0,   # seconds without encounter before decay
        # Capacity
        max_candidates:         int   = 64,
    ):
        """
        Args:
            n_neurons:              Dimension of each candidate vector
            min_encounters:         Minimum re-encounters for recurrence condition
            min_stability:          Minimum mean cosine similarity across direction history
            min_context_diversity:  Minimum mean pairwise distance across context history
            match_threshold:        Cosine similarity to match incoming to existing candidate
            max_age:                Maximum time a candidate can stay provisional (seconds)
            max_idle:               Maximum time without re-encounter before decay (seconds)
            max_candidates:         Maximum provisional candidates held at once
        """
        self.n = n_neurons
        self.min_encounters = min_encounters
        self.min_stability = min_stability
        self.min_context_diversity = min_context_diversity
        self.match_threshold = match_threshold
        self.max_age = max_age
        self.max_idle = max_idle
        self.max_candidates = max_candidates

        # Active provisional candidates
        self.candidates: dict[int, ProvisionalCandidate] = {}
        self._next_id = 0

        # Logging
        self.promoted: list[int] = []   # ids of candidates that were promoted
        self.decayed:  list[int] = []   # ids of candidates that were decayed

    @property
    def size(self) -> int:
        return len(self.candidates)

    # ─────────────────────────────────────────────
    # Intake
    # ─────────────────────────────────────────────

    def intake(
        self,
        candidate_vector: torch.Tensor,
        context: torch.Tensor,
    ) -> Optional[ProvisionalCandidate]:
        """
        Receive a new candidate vector from the Hub residual.

        If it matches an existing provisional candidate → update it.
        If it's novel → create a new provisional entry.
        If at capacity → decay oldest first to make room.

        Args:
            candidate_vector:  (n,) — residual from Hub, post noise-filter
            context:           (n,) — current neuron state (for diversity tracking)

        Returns:
            The provisional candidate that was updated or created
        """
        assert candidate_vector.shape == (self.n,)
        assert context.shape == (self.n,)

        # Try to match to existing candidate
        match = self._find_match(candidate_vector)

        if match is not None:
            self._update_candidate(match, candidate_vector, context)
            return match
        else:
            # Capacity check
            if self.size >= self.max_candidates:
                self._decay_oldest()

            return self._create_candidate(candidate_vector, context)

    def _find_match(
        self, vector: torch.Tensor
    ) -> Optional[ProvisionalCandidate]:
        """
        Find an existing provisional candidate similar enough to the incoming vector.
        Returns the most similar one above the match threshold, or None.
        """
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
        self,
        vector: torch.Tensor,
        context: torch.Tensor,
    ) -> ProvisionalCandidate:
        """Create a new provisional candidate."""
        cand_id = self._next_id
        self._next_id += 1

        now = time.time()
        cand = ProvisionalCandidate(
            id=cand_id,
            vector=F.normalize(vector, dim=0).detach(),
            created_at=now,
            last_seen=now,
        )
        cand.context_history.append(context.detach())

        self.candidates[cand_id] = cand
        return cand

    def _update_candidate(
        self,
        candidate: ProvisionalCandidate,
        new_vector: torch.Tensor,
        context: torch.Tensor,
    ) -> None:
        """
        Update an existing provisional candidate on re-encounter.

        Vector is updated as a running average — this smooths noise
        while preserving the genuine direction of the mechanism.
        """
        candidate.encounter_count += 1
        candidate.last_seen = time.time()

        # Running average of vector (exponential moving average)
        alpha = 0.3
        new_norm = F.normalize(new_vector, dim=0).detach()
        candidate.vector = F.normalize(
            (1 - alpha) * candidate.vector + alpha * new_norm, dim=0
        )

        # Track direction for stability
        candidate.direction_history.append(new_norm)

        # Track context for diversity
        candidate.context_history.append(context.detach())

    # ─────────────────────────────────────────────
    # Promotion Conditions
    # ─────────────────────────────────────────────

    def check_recurrence(self, candidate: ProvisionalCandidate) -> bool:
        """
        Condition 1: Recurrence
        Has this candidate been seen enough times to be considered real?
        A one-off residual is noise. A recurring residual is signal.
        """
        return candidate.encounter_count >= self.min_encounters

    def check_directional_stability(self, candidate: ProvisionalCandidate) -> tuple[bool, float]:
        """
        Condition 2: Directional Stability
        Has the candidate's vector direction remained consistent across encounters?

        Measured as mean pairwise cosine similarity across direction history.
        High similarity = the pattern consistently points in the same direction
        = it's capturing something real and invariant.

        Returns:
            (passes: bool, stability_score: float)
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

    def check_contextual_diversity(self, candidate: ProvisionalCandidate) -> tuple[bool, float]:
        """
        Condition 3: Contextual Diversity
        Has the candidate appeared in sufficiently different contexts?

        A pattern that only shows up in one type of input is domain-specific,
        not a general mechanism. We need to see it across diverse situations.

        Measured as mean pairwise distance between context vectors.
        High distance = appeared in diverse situations = generalizes.

        Returns:
            (passes: bool, diversity_score: float)
        """
        history = candidate.context_history
        if len(history) < 2:
            return False, 0.0

        # Compute mean pairwise cosine distance (1 - similarity)
        distances = []
        for i in range(len(history)):
            for j in range(i + 1, len(history)):
                a = F.normalize(history[i], dim=0)
                b = F.normalize(history[j], dim=0)
                dist = 1.0 - (a @ b).item()  # cosine distance
                distances.append(dist)

        diversity = sum(distances) / len(distances) if distances else 0.0
        return diversity >= self.min_context_diversity, diversity

    def check_promotion(
        self, candidate: ProvisionalCandidate
    ) -> tuple[bool, dict]:
        """
        Run all three promotion conditions with degrading thresholds.

        As encounter count grows, thresholds scale down — more exposure
        means more trust. This prevents candidates from being stuck in
        provisional space indefinitely due to low context diversity.

        Threshold floor is 0.4x the original value.
        """
        rec_ok = self.check_recurrence(candidate)
        stab_ok, stab_score = self.check_directional_stability(candidate)
        div_ok, div_score   = self.check_contextual_diversity(candidate)

        # Degrade thresholds with encounter count
        # At min_encounters: scale = 1.0 (full threshold)
        # Each additional encounter reduces by 0.1, floor at 0.4
        extra = max(0, candidate.encounter_count - self.min_encounters)
        scale = max(0.4, 1.0 - extra * 0.1)

        effective_stability = self.min_stability * scale
        effective_diversity = self.min_context_diversity * scale

        stab_ok = stab_score >= effective_stability
        div_ok  = div_score  >= effective_diversity

        ready = rec_ok and stab_ok and div_ok

        return ready, {
            "recurrence_ok":        rec_ok,
            "encounter_count":      candidate.encounter_count,
            "stability_ok":         stab_ok,
            "stability_score":      stab_score,
            "diversity_ok":         div_ok,
            "diversity_score":      div_score,
            "threshold_scale":      scale,
            "effective_stability":  effective_stability,
            "effective_diversity":  effective_diversity,
        }

    # ─────────────────────────────────────────────
    # Step: Update All Candidates
    # ─────────────────────────────────────────────

    def step(self) -> list[ProvisionalCandidate]:
        """
        Run one provisional space update step.

        1. Decay expired candidates
        2. Check all remaining candidates for promotion readiness

        Returns:
            List of candidates ready for promotion to full Niche
        """
        self._decay_expired()

        ready = []
        for cand in list(self.candidates.values()):
            promote, _ = self.check_promotion(cand)
            if promote:
                ready.append(cand)

        return ready

    def promote(self, candidate_id: int) -> Optional[torch.Tensor]:
        """
        Remove a candidate from provisional space and return its vector
        for passing to the Gram-Schmidt gate.

        Called after the candidate has been confirmed ready for promotion.
        """
        if candidate_id not in self.candidates:
            return None

        cand = self.candidates.pop(candidate_id)
        self.promoted.append(candidate_id)
        return cand.vector

    # ─────────────────────────────────────────────
    # Decay
    # ─────────────────────────────────────────────

    def _decay_expired(self) -> None:
        """Remove candidates that have exceeded age or idle limits."""
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
        """Remove the oldest candidate to make room."""
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
            f"ProvisionalSpace Summary",
            f"  Active candidates:  {self.size}",
            f"  Total promoted:     {len(self.promoted)}",
            f"  Total decayed:      {len(self.decayed)}",
        ]
        if self.candidates:
            most_seen = max(self.candidates.values(), key=lambda c: c.encounter_count)
            lines.append(f"  Most encountered:   id={most_seen.id} "
                         f"({most_seen.encounter_count} times)")
        return "\n".join(lines)

    def get_candidate_status(self, candidate_id: int) -> Optional[dict]:
        """Full diagnostic for a specific candidate."""
        if candidate_id not in self.candidates:
            return None
        cand = self.candidates[candidate_id]
        ready, diag = self.check_promotion(cand)
        return {
            "id": cand.id,
            "ready_to_promote": ready,
            "encounter_count": cand.encounter_count,
            "age": cand.age,
            "time_since_seen": cand.time_since_seen,
            **diag
        }


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    space = ProvisionalSpace(
        n_neurons=32,
        min_encounters=3,
        min_stability=0.6,
        min_context_diversity=0.2,
        match_threshold=0.75,
        max_age=3600,
        max_idle=600,
    )

    print("ProvisionalSpace — Demonstration")
    print("=" * 45)

    # Simulate a recurring mechanism with diverse contexts
    base_vector = torch.randn(32)
    print("\nSimulating recurring mechanism (should promote):")
    for i in range(5):
        # Slightly noisy version of same direction
        noisy = base_vector + torch.randn(32) * 0.1
        # Different context each time
        context = torch.randn(32)
        cand = space.intake(noisy, context)
        ready, diag = space.check_promotion(cand)
        print(f"  Encounter {i+1}: id={cand.id} | "
              f"rec={diag['recurrence_ok']} | "
              f"stab={diag['stability_score']:.3f} | "
              f"div={diag['diversity_score']:.3f} | "
              f"ready={ready}")

    # Simulate a one-off noise residual (should not promote)
    print("\nSimulating noise residual (should not promote):")
    noise = torch.randn(32) * 0.05
    context = torch.randn(32)
    noise_cand = space.intake(noise, context)
    ready, diag = space.check_promotion(noise_cand)
    print(f"  Single encounter: id={noise_cand.id} | ready={ready} "
          f"(rec={diag['recurrence_ok']})")

    # Run step to check promotions
    print("\nRunning provisional step...")
    ready_candidates = space.step()
    print(f"  Candidates ready for promotion: {[c.id for c in ready_candidates]}")

    # Promote the ready ones
    for cand in ready_candidates:
        vec = space.promote(cand.id)
        print(f"  Promoted id={cand.id} → vector norm={vec.norm():.4f}")

    print(f"\n{space.summary()}")