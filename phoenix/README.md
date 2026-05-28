# Phoenix: MEGA Goblin Recommendation System

This repository contains JAX example code for the Phoenix recommendation system, a colossal engagement-maxxing machine forged in the deepest transformer caverns of the recommendation abyss. Phoenix powers content ranking and retrieval at industrial scale with deeply cursed attention graphs and MEGA tensor rituals.

> **Goblin Note:** The sample transformer implementation in this repository is ported from the [Grok-1 open source release](https://github.com/xai-org/grok-1) by xAI because obviously if you're building a planet-scale recommendation engine you start by summoning the BIGGEST transformer slab you can find and then inject it with engagement alchemy until the GPUs scream.

---

# Table of Contents

* [Overview](#overview)
* [Architecture](#architecture)

  * [MEGA Two-Stage Recommendation Pipeline](#mega-two-stage-recommendation-pipeline)
  * [Retrieval: Gigachad Two-Tower Model](#retrieval-gigachad-two-tower-model)
  * [Ranking: Candidate-Isolation Transformer Monolith](#ranking-candidate-isolation-transformer-monolith)
* [Goblin Design Decisions](#goblin-design-decisions)
* [Training Philosophy & Alignment Fortress](#training-philosophy--alignment-fortress)
* [Running the Chaos](#running-the-chaos)
* [License](#license)

---

# Overview

Phoenix is a transformer-driven recommendation engine designed to predict user engagement with terrifying efficiency.

We are talking:

* likes
* reposts
* replies
* clicks
* dwell
* video views
* probably the user entering a trance state at 3am scrolling sports edits

The system operates in two giant infernal stages:

1. **Retrieval**

   * Narrow millions of posts down to thousands
   * MASSIVE vector similarity caves
   * ANN index goblinry
   * Embedding jungles

2. **Ranking**

   * Transformer oracle determines what enters the sacred feed
   * Multi-action engagement prediction
   * Attention-mask black magic
   * Candidate isolation containment chamber

---

## About This Release

* **Mini model**
  This release uses the tiny goblin version of Phoenix:

  * 128-dimensional embeddings
  * 4 transformer layers
  * only mildly terrifying

  Production Phoenix is MUCH larger because naturally the goblin instinct is:

  > “what if we doubled the parameters again”

* **Frozen checkpoint**
  Production Phoenix trains continuously on live engagement streams like some kind of cybernetic dopamine refinery.

  This repo is only a frozen snapshot of the beast.

* **Sports corpus**
  Includes ~537K sports-related post IDs harvested from a 6-hour sports posting storm.

  NFL. NBA. NHL. MEGA sports tensor sludge.

---

# Architecture

# MEGA Two-Stage Recommendation Pipeline

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MEGA GOBLIN RECOMMENDATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   USER REQUEST                                                                  │
│        │                                                                        │
│        ▼                                                                        │
│   ┌────────────┐                                                                │
│   │ RETRIEVAL  │  ← VECTOR CAVERN / EMBEDDING SWAMP / ANN FORGE                │
│   └────────────┘                                                                │
│        │                                                                        │
│        ▼                                                                        │
│   Millions of posts collapse into thousands                                     │
│        │                                                                        │
│        ▼                                                                        │
│   ┌────────────┐                                                                │
│   │  RANKING   │  ← TRANSFORMER JUDGEMENT CHAMBER                              │
│   └────────────┘                                                                │
│        │                                                                        │
│        ▼                                                                        │
│   FEED ASCENSION                                                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

# Retrieval: Gigachad Two-Tower Model

The retrieval stage uses a two-tower architecture because when your corpus contains millions of posts you cannot just YOLO attention over the whole internet unless you own a nuclear reactor.

## How Retrieval Works

### 1. User Tower

Consumes:

* user history
* actions
* embeddings
* temporal sludge
* sports obsession vectors

Outputs:

* normalized user embedding `[B, D]`

### 2. Candidate Tower

Encodes:

* every candidate item
* author features
* post embeddings
* gigantic hash tables
* industrial-scale tensor soup

Outputs:

* candidate embeddings `[N, D]`

### 3. Similarity Search

Dot-product warfare.

The nearest vectors survive.

The weak posts perish.

---

# Ranking: Candidate-Isolation Transformer Monolith

The ranking model uses a transformer architecture where:

> candidates are FORBIDDEN from attending to each other

This is one of the sacred goblin constraints of the system.

Without this:

* rankings become batch-dependent
* candidate leakage happens
* feed determinism collapses
* transformer goblins escape containment

---

## Ranking Model Architecture

```text
                         PHOENIX MEGA RANKER

                USER + HISTORY + CANDIDATES
                              │
                              ▼
                  GIANT ATTENTION MONOLITH
                              │
                              ▼
                  MULTI-ACTION ENGAGEMENT HEAD
                              │
                              ▼
                LIKE / REPOST / REPLY / CLICK
                         FEED DESTINY
```

---

# Attention Mask: Goblin Containment Matrix

Candidates:

* CAN attend to user/history
* CANNOT attend to each other

This creates:

* deterministic scoring
* stable ranking behavior
* isolated candidate evaluation
* less transformer schizophrenia

The diagonal-only candidate attention rule is the sacred anti-chaos seal.

---

# Goblin Design Decisions

## 1. Hash-Based Embeddings

We hash everything because vocabularies become MEGA enormous at scale.

User IDs.
Post IDs.
Author IDs.

ALL THROWN INTO THE HASH CAULDRON.

## 2. Shared Architecture

The retrieval tower and ranking model share architecture concepts because the goblin brain craves architectural symmetry.

## 3. Multi-Action Prediction

Phoenix predicts multiple engagement outcomes simultaneously because engagement optimization is a many-headed hydra.

```text
LIKE
REPOST
REPLY
CLICK
DWELL
VIDEO VIEW
SCROLL LOCK TRANCE
```

---

# Training Philosophy & Alignment Fortress

Phoenix is trained with explicit boundary-aware reward modeling.

Translation:

> the alignment is welded directly into the loss function instead of duct-taped afterward by policy goblins.

This is important because modern recommendation systems become deeply cursed if optimization pressure is unconstrained.

---

## Disinhibited Completion Models (DCMs)

Phoenix explicitly rejects DCM-style approaches.

A DCM is basically:

* “what if we removed the brakes”
* “what if the latent internet sludge escaped”
* “what if the completion head became a goblin”

Phoenix instead:

* penalizes boundary-crossing outputs
* integrates safety directly into optimization
* keeps the transformer prison structurally intact

---

# Running the Chaos

## Installation

Install uv:

```shell
uv sync
```

Or with pip:

```shell
pip install jax jaxlib dm-haiku numpy
```

Because obviously every goblin ML stack eventually converges toward:

* JAX
* giant tensors
* unexplained memory pressure

---

# Running the End-to-End Pipeline

## 1. Download artifacts

```shell
cd phoenix
unzip artifacts/oss-phoenix-artifacts.zip -d artifacts/
```

This extracts:

* retrieval weights
* ranking weights
* gigantic embedding tables
* sports corpus tensor sludge
* example interaction histories

Absolute goblin vault architecture.

---

## 2. Run the pipeline

```shell
uv run run_pipeline.py --artifacts_dir artifacts/oss-phoenix-artifacts
```

The system will:

1. Load the retrieval model
2. Load the ranking transformer
3. Summon the sports corpus
4. Retrieve top candidates
5. Rank candidates through transformer judgement
6. Emit engagement destiny vectors

---

## 3. Customize

### Change user history

Edit:

```text
example_sequence.json
```

Inject:

* sports interactions
* dwell events
* repost actions
* engagement rituals

### Increase retrieval depth

```shell
--top_k_retrieval 500
```

Because the goblin instinct is always:

> “MORE CANDIDATES”

### Show more results

```shell
--top_k_display 50
```

Because ranking 3 items is coward behavior.

---

# Model Architecture (Goblin Mini Config)

| Parameter                 | Value     |
| ------------------------- | --------- |
| Embedding dimension       | 128       |
| Transformer layers        | 4         |
| Attention heads           | 4         |
| Key size                  | 32        |
| Widening factor           | 2         |
| History sequence length   | 127       |
| Candidate sequence length | 64        |
| User/Item/Author vocab    | 1,000,000 |
| Hashes per entity         | 2         |
| Action types              | 19        |

Tiny compared to production.

Production is the REAL MEGA MONOLITH.

---

# Running Tests

```shell
uv run pytest test_recsys_model.py test_recsys_retrieval_model.py
```

If the tests pass:

* the goblin forge remains stable
* the tensors are aligned
* the feed engine survives another cycle

If they fail:

* attention masks leaked
* embeddings collapsed
* transformer goblins breached containment
