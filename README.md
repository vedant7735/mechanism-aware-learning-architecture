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

The Hub is a dual-signal coordination module — the most novel component of the architecture.

**Signal 1 — Change Vector (per neuron):**
Tracks what changed for each neuron, by how much, and why. Each neuron's change is decomposed into contributions from existing Niche activations plus a residual. The residual is unexplained change — the raw material for potential new Niche formation. Every change in the network has a structured, attributable explanation.

**Signal 2 — Sparse Correlation Structure:**
A compressed representation of inter-neuron co-change relationships. Not a full n×n matrix (which scales quadratically) — a sparse approximation tracking only the most significant relational dynamics.

The Hub's primary function is **weight modulation**: adjusting the mechanism-factorized state based on these two signals — not raw error minimization. Niche selection operates on both signals: what changed, and what changed together.

### Niche Formation Gate

New mechanisms are admitted to the library only if they satisfy all four conditions:

1. **Generalize** across contexts — not specific to one observation
2. **Stable** under regime shift — not a surface correlation
3. **Add information** — explain dynamics not covered by existing Niches
4. **Decorrelated** — cosine similarity to all existing Niches is below threshold

The gate is implemented via **Gram-Schmidt orthogonalization**: the candidate Niche vector is projected against all existing columns, existing contributions are subtracted out, and only the genuinely new residual is evaluated. If it passes, it is admitted as a new column. If not, it merges, compresses, or is discarded.

Existing Niches are not passive — they actively shape what new Niches become, by defining what is already explained. The library builds on itself.

### Hierarchical Mechanism Library

Mechanisms are organized as:
- **Abstraction hierarchy**: general mechanisms at the top, increasingly specific below
- **Similarity-linked graph**: mechanisms sharing structural properties are linked via correlation pointers

This enables fast recall, compositional reasoning, redundancy control, and natural abstraction depth.

### Foundation Model + Task Models

```
Foundation Model  →  mechanism knowledge  (the brain)
Task-Specific Models  →  domain action  (the hands)
```

The Foundation Model is trained once and reused. Task models inherit its Niche library without relearning mechanism-understanding from scratch — they learn only what actions to take within their domain.

---

## Mechanism Lifecycle

**Early learning:** Small m, broad general mechanisms, coarse approximations.

**As exposure increases:** New Niches emerge when existing ones cannot explain a pattern invariantly. Abstraction resolution increases. Pattern identification becomes faster and more precise.

**After stabilization:** Mechanisms are distilled into compact representations — small neural modules, then low-rank dynamics. Learning is expensive early. Intelligence becomes cheap over time.

This mirrors human expertise: effortful reasoning as a novice, efficient recognition as an expert.

---

## Failure Mode Reframing

Failures are structurally suppressed and — critically — traceable. When the system fails, it points to a specific cause:

- **Incomplete mechanism** — the Niche library does not yet cover the relevant dynamic
- **Wrong selection** — the Hub activated the incorrect Niche for the context  
- **Insufficient awareness** — m is too small to distinguish between similar dynamics

Failure becomes a learning signal for the architecture itself, not just for the model's weights.

---

## Project Structure

```
maia/
│
├── core/
│   ├── hub.py              # Central Awareness Hub (dual-signal system)
│   ├── niche.py            # Mechanism storage, updates, library management
│   └── formation_gate.py   # Gram-Schmidt orthogonalization + invariance checks
│
├── experiments/
│   └── toy_dynamics.py     # Simple mechanism discovery tasks
│
├── utils/
│   └── similarity.py       # Cosine similarity, correlation utilities
│
├── docs/
│   └── spec_v3.docx        # Full formal specification
│
└── main.py
```

---

## Research Hypotheses

These are the falsifiable claims the architecture is built to test:

1. **Mechanism scaling** — Niche count m scales favorably with model size, producing measurable gains in learning efficiency and generalization
2. **OOD generalization** — Mechanism selection generalizes to out-of-distribution inputs better than pattern-memorizing baselines at equivalent parameter counts
3. **Data efficiency** — Task models built on the Foundation Model require less domain-specific data than equivalent standalone models
4. **Complexity robustness** — Performance degradation as task complexity increases is lower than in pattern-memorizing architectures
5. **Failure traceability** — Errors can be attributed to specific mechanism gaps rather than global statistical noise
6. **Orthogonality maintenance** — The Gram-Schmidt gate keeps the Niche library decorrelated as m grows, preventing redundancy creep

---

## Roadmap

- [x] Formal specification (v3.0)
- [ ] Toy prototype — Niche formation under Gram-Schmidt gate
- [ ] Toy prototype — Hub dual-signal update
- [ ] Toy prototype — Niche selection generalizing to held-out patterns
- [ ] Formal mathematical specification (update rules, scaling proofs)
- [ ] OOD generalization experiments vs. baselines
- [ ] Research paper (target: IEEE TNNLS / NeurIPS / ICLR)

> **Currently building:** A minimal demonstration implementation to validate the core Niche formation and selection mechanism at small scale before moving to full architecture experiments.

---

## Background

This architecture draws inspiration from — and is distinct from — the following research areas:

- **Causal representation learning** — extracting invariant structure; MAIA extends toward dynamic mechanisms rather than static causal graphs
- **Meta-learning** — generalizing through structure; MAIA uses explicit labeled Niches rather than gradient-based meta-optimization
- **Continual learning** — avoiding catastrophic forgetting; MAIA addresses this through mechanism isolation and orthogonalization
- **Neuromodulated learning** — global signals modulating learning dynamics; Niches are a more explicit, hierarchical, multi-channel version
- **Mixture of Experts** — multi-channel selection; MoE selects execution paths, MAIA selects change-understanding frameworks
- **Curriculum learning** — structured exposure ordering; MAIA requires this by architectural design

---

## Tech Stack

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib

---

## Status

**Active research.** Formal specification complete (v3.0). Toy demonstration under development. This is independent research conceived and developed from first principles.

---

## License

MIT
