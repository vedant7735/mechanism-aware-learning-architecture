# 🧠 MAIA — Mechanism-Aware Intelligence Architecture

![Status](https://img.shields.io/badge/status-active%20research-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Stage](https://img.shields.io/badge/stage-prototype-orange)

> **Intelligence scales through mechanism discovery and reuse — not through data accumulation.**
>
> Data is leverage for extracting structure. Mechanisms become leverage over data.

---

## What is MAIA?

MAIA is a research architecture that rethinks how learning systems build intelligence.

Current deep learning systems scale by fitting statistical correlations across massive datasets. They are good at pattern matching. They are brittle outside their training distribution, and when they fail, they fail opaquely.

MAIA instead treats data as **exposure** — raw material for discovering invariant generative mechanisms. Once a mechanism is internalized, the model does not need to re-encounter similar data to handle new situations. It applies the mechanism. This is how human intelligence actually works: not lookup, not nearest-neighbor retrieval — mechanism identification and reuse.

The result is a system designed to generalize through structural understanding, degrade gracefully with complexity, and fail traceably when it does fail.

---

## Core Thesis

```
Features are observations. Niches are laws.
A model with features has seen a lot. A model with Niches understands something.
```

| | Pattern-Memorizing Systems | MAIA |
|---|---|---|
| **What data does** | Provides examples to fit | Provides exposure to extract mechanisms |
| **What scales intelligence** | More data, more parameters | More mechanism discovery |
| **Generalization method** | Statistical proximity | Mechanism identification |
| **Failure mode** | Opaque — statistical error | Traceable — specific mechanism gap |
| **OOD performance** | Degrades sharply | Designed to degrade gracefully |

---

## Two Defining Properties

### Structural Awareness
The model distinguishes types of dynamics, not just surface patterns. It treats linear drift differently from oscillation. It represents mechanism A separately from mechanism B. It does not collapse everything into one blended statistical representation.

**Measured by:** lower residual after mechanism decomposition, cleaner Niche specialization, better OOD performance when structure reappears.

### Memory Awareness
The model stores a discovered mechanism, recalls it later, and reapplies it without relearning from scratch. Previously seen dynamics are recognized and handled through existing Niches, not re-derived.

**Measured by:** fewer new Niches formed over time, faster adaptation to recurring dynamics, lower data requirement for previously encountered patterns.

These two properties together constitute what it means for a system to be mechanistically intelligent rather than statistically capable. Current architectures optimize for neither explicitly. MAIA does.

---

## Key Concepts

### Niches (Mechanism Channels)

A **Niche** is a labeled, internalized framework for understanding a specific invariant change-dynamic. It is not a learned feature. The distinction is precise:

- A **feature** is a statistical regularity discovered to minimize loss. It is data-dependent, numerous, and opaque.
- A **Niche** is an invariant generative mechanism promoted by passing a strict formation gate. It is stable across regimes, sparse, and attributable.

Every Niche is a feature. Not every feature qualifies as a Niche.

The number of active Niches, **m**, scales with model size — larger models support richer mechanism libraries, providing finer-grained contextual understanding. This is the formal basis of the **awareness spectrum**.

### Awareness

Awareness is defined operationally — not philosophically:

> The ability to recognize a pattern and recall the correct underlying mechanism that governs it.

This explicitly excludes output lookup and pattern matching. It explicitly includes mechanism identification and reuse. Higher awareness means a richer, deeper, more interconnected mechanism library. It is continuous, not binary.

### The Mechanism-Factorized State

The model maintains knowledge as a structured two-dimensional representation:

- **n** = neurons / local pattern processors (rows — encode observations)
- **m** = Niches (columns — encode invariant change-dynamics)

Patterns are interpreted across Niche columns rather than directly mapped to outputs. Output is derived from mechanism activation, not pattern proximity. Communication between components uses **vectors, not matrices** — keeping the system efficient and well-defined.

---

## Architecture

### Central Awareness Hub

The Hub is a dual-signal coordination module operating at O(n) complexity — not O(n²) like attention mechanisms.

**Signal 1 — Change Vector (per neuron):**
Tracks what changed for each neuron, by how much, and why. Each neuron's change is decomposed into contributions from existing Niche activations plus a residual. The residual is unexplained change — candidate material for the intake pipeline.

**Signal 2 — Sparse Correlation Structure:**
A compressed representation of inter-neuron co-change relationships. Not a full n×n matrix — a sparse approximation tracking only the most significant relational dynamics.

The Hub normalizes neuron states before computing change vectors, making mechanism detection **scale-invariant**. A linearly growing signal and a unit signal produce the same directional change — the same mechanism.

### Residual Classification Layer (v5.0)

A band-stop filter that routes incoming residuals to the correct intake pathway:

- **Low variation** → Monotonic Detector (consistent, context-independent mechanisms)
- **Mid variation** → Provisional Space (context-dependent mechanisms needing evidence)
- **High variation** → Discard (noise)

### Monotonic Detector (v5.0)

Fast-track intake for mechanisms that are too consistent to satisfy the Provisional Space's diversity condition but too persistent to be noise. A perfectly consistent residual is its own evidence — it does not need diverse contexts to prove generality.

**Promotion conditions:** recurrence + directional stability + confirmed low variation. No diversity condition.

### Provisional Space (v4.0)

Standard intake for context-dependent mechanisms. Candidates accumulate evidence across encounters before being considered for promotion.

**Promotion conditions:** recurrence + directional stability + contextual diversity. Thresholds degrade with encounter count — more exposure means more trust.

### Niche Formation Gate

New mechanisms are admitted to the library only if they satisfy all four conditions:

1. **Generalize** across contexts — not specific to one observation
2. **Stable** under regime shift — not a surface correlation
3. **Add information** — explain dynamics not covered by existing Niches
4. **Decorrelated** — cosine similarity to all existing Niches is below threshold

Implemented via **Gram-Schmidt orthogonalization**. Existing Niches are not passive — they actively shape what new Niches become by defining what is already explained.

### Hierarchical Mechanism Library

Mechanisms are organized as an abstraction hierarchy (general → specific) with similarity-linked graph connections. Enables fast recall, compositional reasoning, and redundancy control.

### Foundation Model + Task Models

```
Foundation Model  →  mechanism knowledge  (the brain)
Task-Specific Models  →  domain action  (the hands)
```

The Foundation Model is trained once and reused. Task models inherit its Niche library without relearning mechanism-understanding from scratch.

---

## Mechanism Lifecycle

**Early learning:** Small m, broad general mechanisms. Many candidates enter intake pathways. Few graduate immediately.

**As exposure increases:** Candidates accumulate evidence. Those with genuine invariance graduate. Those without decay. The library grows slowly but cleanly.

**After stabilization:** Mechanisms distilled into compact representations. Learning is expensive early. Intelligence becomes cheap over time.

This mirrors human expertise: effortful reasoning as a novice, efficient recognition as an expert.

---

## Failure Modes (Six Diagnosable States)

Failures are structurally suppressed and traceable. When the system fails, it points to a specific cause:

- **Incomplete mechanism** — no Niche or candidate covers this context
- **Provisional pending** — matches provisional candidate, still accumulating evidence
- **Monotonic pending** — matches monotonic candidate, accumulating recurrence
- **Wrong selection** — Hub activated incorrect Niche; composition may help
- **Insufficient awareness** — m too small to distinguish similar dynamics
- **Weight collapse** — relevant Niche exists but weight has decayed below threshold

Failure becomes a learning signal for the architecture itself, not just for model weights.

---

## Empirical Results (v5.0 Prototype)

All five core hypotheses validated in controlled toy experiments:

| Experiment | Result |
|---|---|
| Mechanism discovery & reuse | ✓ Explanation ratio 0.017 → 0.472. Niche formed at step 14, reused thereafter. |
| Distinct Niches per dynamic | ✓ Linear reappearance: 0 new Niches formed. Existing Niche reused. |
| Composition on mixed dynamics | ✓ 75% composition usage on mixed dynamics. |
| Failure traceability | ✓ All 5 test cases correctly diagnosed — familiar and novel contexts. |
| Awareness spectrum scaling | ✓ Explanation ratio monotonically increases with Niche cap. |

**Key architectural finding:** Scale-invariant change detection was necessary for consistent mechanism detection across dynamics with growing magnitude. Linear dynamics require the Monotonic Detector pathway — the Provisional Space's diversity condition correctly rejects them since they are context-independent by nature. This distinction between context-dependent and context-independent mechanisms is an empirically confirmed architectural property.

---

## Project Structure

```
MAIA/
│
├── experiments/
│   ├── hub.py                  # Central Awareness Hub (dual-signal, scale-invariant)
│   ├── formation_gate.py       # Gram-Schmidt orthogonalization + invariance checks
│   ├── niche.py                # Mechanism storage, selection, composition, hierarchy
│   ├── provisional.py          # Provisional Space — context-dependent intake
│   ├── monotonic_detector.py   # Monotonic Detector + Residual Classifier
│   ├── main.py                 # End-to-end MAIA system (v5.0)
│   └── toy_dynamics.py         # Controlled experiment suite (5 experiments)
│
├── utils/
│   └── similarity.py           # Cosine similarity, correlation utilities
│
├── docs/
│   └── architecture_spec_v5.docx   # Full formal specification
│
└── README.md
```

---

## Research Hypotheses (Ten Claims)

1. **Mechanism scaling** — Niche count m scales favorably with model size ✓ *confirmed*
2. **OOD generalization** — Mechanism selection generalizes better than pattern-matching baselines
3. **Data efficiency** — Task models require less domain-specific data than standalone models
4. **Complexity robustness** — Performance degrades more slowly than pattern-memorizing architectures
5. **Failure traceability** — Errors attributed to specific mechanism gaps ✓ *confirmed*
6. **Orthogonality maintenance** — Gram-Schmidt gate keeps library decorrelated as m grows
7. **Provisional space efficiency** — Three-state system produces cleaner libraries than binary gate ✓ *confirmed*
8. **Curriculum tolerance** — Dual intake pathway reduces curriculum sensitivity ✓ *confirmed*
9. **Attribution traceability** — Mechanism formation traceable to specific neurons via impact matrix
10. **Weight-driven pruning efficiency** — Competitive dynamics produce leaner libraries than fixed compression

---

## Roadmap

- [x] Formal specification (v5.0)
- [x] Central Awareness Hub — dual-signal, scale-invariant
- [x] Niche Formation Gate — Gram-Schmidt orthogonalization
- [x] Niche Library — selection, composition, hierarchy
- [x] Provisional Space — three-state epistemic system
- [x] Monotonic Detector — dual intake pathway with band-stop classifier
- [x] Experiment suite — 5 controlled experiments
- [x] All 5 hypotheses validated on toy prototype
- [ ] NeuronID attribution system — Neuron × Niche impact matrix
- [ ] Weighted Niche dynamics — competitive weight updates
- [ ] Niche embedding space — base Niche anchors
- [ ] Curriculum-driven training data pipeline
- [ ] Formal mathematical specification
- [ ] OOD generalization experiments vs. baselines
- [ ] Research paper (target: IEEE TNNLS / NeurIPS / ICLR)

---

## Background

This architecture draws inspiration from — and is distinct from — the following research areas:

- **Causal representation learning** — MAIA extends toward dynamic mechanisms rather than static causal graphs
- **Meta-learning** — MAIA uses explicit labeled Niches rather than gradient-based meta-optimization
- **Continual learning** — mechanism isolation and orthogonalization address catastrophic forgetting
- **Neuromodulated learning** — Niches are a more explicit, hierarchical, multi-channel version
- **Mixture of Experts** — MoE selects execution paths; MAIA selects change-understanding frameworks
- **Memory consolidation** — the dual intake pathway mirrors biological memory consolidation
- **Word embeddings** — Niche embedding space draws structural inspiration from semantic spaces
- **Credit assignment** — NeuronID attribution formalizes credit assignment at the mechanism level
- **Transformer attention** — MAIA achieves global coordination at O(n) vs O(n²)

---

## Tech Stack

- Python 3.10+
- PyTorch
- NumPy

---

## Status

**Active research.** Formal specification v5.0 complete. Toy prototype running with all 5 hypotheses validated. Next: NeuronID attribution system and weighted Niche dynamics.

This is independent research conceived and developed from first principles.

---

## License

MIT