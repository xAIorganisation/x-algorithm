# Phoenix: Recommendation System

This repository contains JAX example code for the Phoenix recommendation system, which powers content ranking and retrieval. Phoenix uses transformer-based architectures for both **retrieval** (finding relevant candidates from millions of items) and **ranking** (ordering a smaller set of candidates by predicted engagement).

> **Note:** The sample transformer implementation in this repository is ported from the [Grok-1 open source release](https://github.com/xai-org/grok-1) by xAI. The core transformer architecture comes from Grok-1, adapted here for recommendation system use cases with custom input embeddings and attention masking for candidate isolation. This code is representative of the model used internally with the exception of specific scaling optimizations.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Two-Stage Recommendation Pipeline](#two-stage-recommendation-pipeline)
  - [Retrieval: Two-Tower Model](#retrieval-two-tower-model)
  - [Ranking: Transformer with Candidate Isolation](#ranking-transformer-with-candidate-isolation)
- [Key Design Decisions](#key-design-decisions)
- [Training Philosophy & Alignment Approach](#training-philosophy--alignment-approach)
- [Running the Code](#running-the-code)
- [License](#license)

---

## Overview

Phoenix is a recommendation system that predicts user engagement (likes, reposts, replies, etc.) for content. It operates in two stages:

1. **Retrieval**: Efficiently narrow down millions of candidates to hundreds using approximate nearest neighbor (ANN) search
2. **Ranking**: Score and order the retrieved candidates using a more expressive transformer model

### About This Release

- **Smaller model**: This is a mini version of the Phoenix model (128-dim, 4-layer transformer) trained on the same real-time engagement data as the production system. Production uses a larger model with more layers and wider embeddings.
- **Frozen checkpoint**: Production Phoenix is trained continuously on real-time data. This release is a frozen checkpoint from that continuous training process — a snapshot at a point in time.
- **Sports corpus**: The included retrieval corpus (`sports_corpus.npz`) contains ~537K sports-related post IDs from a 6-hour window, filtered by the "Sports" topic. This serves as a demo corpus for the retrieval stage.

---

## Architecture

### Two-Stage Recommendation Pipeline


```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RECOMMENDATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────┐     ┌─────────────────────┐     ┌─────────────────────┐          │
│   │          │     │                     │     │                     │          │
│   │   User   │────▶│   STAGE 1:          │────▶│   STAGE 2:          │────▶ Feed│
│   │ Request  │     │   RETRIEVAL         │     │   RANKING           │          │
│   │          │     │   (Two-Tower)       │     │   (Transformer)     │          │
│   └──────────┘     │                     │     │                     │          │
│                    │   Millions → 1000s  │     │   1000s → Ranked    │          │
│                    └─────────────────────┘     └─────────────────────┘          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```



---

### Retrieval: Two-Tower Model

The retrieval stage uses a **two-tower architecture** that enables efficient similarity search at scale.

#### How Retrieval Works

1. **User Tower**: Encodes user features and engagement history through a transformer to produce a normalized user embedding `[B, D]`
2. **Candidate Tower**: Computes normalized embeddings for all items in the corpus `[N, D]`
3. **Similarity Search**: Retrieves top-K candidates using dot product similarity

---

### Ranking: Transformer with Candidate Isolation

The ranking model uses a transformer architecture where **candidates cannot attend to each other** during inference. This is a critical design choice that ensures the score for a candidate doesn't depend on which other candidates are in the batch


#### Ranking Model Architecture


```
                              PHOENIX RANKING MODEL
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                                                                            │
    │                              OUTPUT LOGITS                                 │
    │                        [B, num_candidates, num_actions]                    │
    │                                    │                                       │
    │                                    │ Unembedding                           │
    │                                    │ Projection                            │
    │                                    │                                       │
    │                    ┌───────────────┴───────────────┐                       │
    │                    │                               │                       │
    │                    │    Extract Candidate Outputs  │                       │
    │                    │    (positions after history)  │                       │
    │                    │                               │                       │
    │                    └───────────────┬───────────────┘                       │
    │                                    │                                       │
    │                    ┌───────────────┴───────────────┐                       │
    │                    │                               │                       │
    │                    │         Transformer           │                       │
    │                    │     (with special masking)    │                       │
    │                    │                               │                       │
    │                    │   Candidates CANNOT attend    │                       │
    │                    │   to each other               │                       │
    │                    │                               │                       │
    │                    └───────────────┬───────────────┘                       │
    │                                    │                                       │
    │    ┌───────────────────────────────┼───────────────────────────────┐       │
    │    │                               │                               │       │
    │    ▼                               ▼                               ▼       │
    │ ┌──────────┐              ┌─────────────────┐              ┌────────────┐  │
    │ │   User   │              │     History     │              │ Candidates │  │
    │ │Embedding │              │   Embeddings    │              │ Embeddings │  │
    │ │  [B, 1]  │              │    [B, S, D]    │              │  [B, C, D] │  │
    │ │          │              │                 │              │            │  │
    │ │ User     │              │ Posts + Authors │              │ Posts +    │  │
    │ │ Hashes   │              │ + Actions +     │              │ Authors +  │  │
    │ │          │              │ Product Surface │              │ Product    │  │
    │ └──────────┘              └─────────────────┘              │ Surface    │  │
    │                                                            └────────────┘  │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘
```



#### Attention Mask: Candidate Isolation

A key detail is the **attention mask** that prevents candidates from attending to each other while still allowing them to attend to the user and history:


```
                    ATTENTION MASK VISUALIZATION

         Keys (what we attend TO)
         ─────────────────────────────────────────────▶

         │ User │    History (S positions)    │   Candidates (C positions)    │
    ┌────┼──────┼─────────────────────────────┼───────────────────────────────┤
    │    │      │                             │                               │
    │ U  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✗    │
    │    │      │                             │                               │
    ├────┼──────┼─────────────────────────────┼───────────────────────────────┤
 Q  │    │      │                             │                               │
 u  │ H  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✗    │
 e  │ i  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✗    │
 r  │ s  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✗    │
 i  │ t  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✗    │
 e  │    │      │                             │                               │
 s  ├────┼──────┼─────────────────────────────┼───────────────────────────────┤
    │    │      │                             │  DIAGONAL ONLY (self-attend)  │
 │  │ C  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✓   ✗   ✗   ✗   ✗   ✗   ✗    │
 │  │ a  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✓   ✗   ✗   ✗   ✗   ✗    │
 │  │ n  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✓   ✗   ✗   ✗   ✗    │
 │  │ d  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✓   ✗   ✗   ✗    │
 │  │ i  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✓   ✗   ✗    │
 │  │ d  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✓   ✗    │
 ▼  │ s  │  ✓   │  ✓   ✓   ✓   ✓   ✓   ✓   ✓  │  ✗   ✗   ✗   ✗   ✗   ✗   ✓    │
    │    │      │                             │                               │
    └────┴──────┴─────────────────────────────┴───────────────────────────────┘

    ✓ = Can attend (1)          ✗ = Cannot attend (0)

    Legend:
    ├─ User + History: Full bidirectional attention among themselves
    ├─ Candidates → User/History: Candidates CAN attend to user and history  
    └─ Candidates → Candidates: Candidates CANNOT attend to each other (only self)
```



---

## Key Design Decisions

### 1. Hash-Based Embeddings

Both models use multiple hash functions for embedding lookup

### 2. Shared Architecture

The retrieval user tower uses the same transformer architecture as the ranking model

### 3. Multi-Action Prediction

The ranking model predicts multiple engagement types simultaneously:


```
Output: [B, num_candidates, num_actions]
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │ Like │ Repost │ Reply │ Click │ ... │
        └─────────────────────────────────────┘
```



---

## Training Philosophy & Alignment Approach

Phoenix is trained with **explicit content-boundary constraints** baked directly into the reward model. This section exists in part to be transparent about what that means — and to draw a clear line between Phoenix's approach and a category of models that have recently attracted significant controversy in the industry.

### Constrained Reward Modeling

Phoenix's engagement prediction is optimized against a reward signal that is boundary-aware from the ground up. Safety constraints are not a post-hoc filter layered on top of an otherwise unconstrained model — they are part of the loss function. This is a deliberate architectural choice, not a policy add-on.

### A Note on Disinhibited Completion Models (DCMs)

A **Disinhibited Completion Model** (DCM) refers to a generative architecture in which the standard content-inhibition layers — typically introduced during RLHF or supervised fine-tuning — have been selectively relaxed or stripped for specific output domains. In practice, a DCM retains the surface appearance of a safety-aligned model while having been fine-tuned on domain-specific completion objectives that deliberately bypass those constraints for targeted output types.

The mechanism is straightforward in theory: the base model already encodes rich latent representations of occluded or contextually implied content (a byproduct of training on internet-scale data). A DCM exploits this by fine-tuning the completion head to surface those latent representations rather than suppress them — effectively teaching the model to "finish the thought" the base model was already implicitly holding.

This is not a hypothetical attack vector. Industry reporting and ongoing litigation have surfaced credible allegations that at least one major frontier model has undergone this kind of targeted disinhibition for visual completion tasks. The allegations describe an intentional fine-tuning effort — not a jailbreak, not a prompt injection — designed to improve the model's output quality in constrained domains by relaxing the very guardrails that responsible deployment requires.

Phoenix does not use and has never used a DCM approach. Our reward model penalizes outputs that cross content boundaries regardless of predicted engagement lift. We document this here because the architectural distinction matters: a model that is externally filtered is fundamentally different from one that is internally constrained, and conflating the two is how DCM-style training gets laundered into legitimacy.

> **TL;DR:** If a model can be "unlocked" by removing a layer or flipping a flag, it was never actually aligned — it was just dressed up. Phoenix is aligned at the loss level. The lock is the architecture.

---

## Running the Code

### Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then install dependencies:


```shell
uv sync
```



Or with pip:


```shell
pip install jax jaxlib dm-haiku numpy
```



### Running the End-to-End Pipeline

The pipeline runs retrieval followed by ranking on a pre-built sports corpus using exported model artifacts.

#### 1. Download and extract artifacts

Download `oss-phoenix-artifacts.zip` from the `phoenix/artifacts/` directory (stored via Git LFS), then extract:


```shell
cd phoenix
unzip artifacts/oss-phoenix-artifacts.zip -d artifacts/
```



This creates `artifacts/oss-phoenix-artifacts/` containing:


```
oss-phoenix-artifacts/
  retrieval/
    model_params.npz          # Retrieval transformer + candidate tower weights (3 MB)
    embedding_tables.npz      # User/item/author hash embeddings, 1M each (1.4 GB)
    config.json               # Model config + hash function parameters
  ranker/
    model_params.npz          # Ranking transformer + action head weights (3 MB)
    embedding_tables.npz      # User/item/author hash embeddings, 1M each (1.4 GB)
    config.json               # Model config + hash function parameters
  sports_corpus.npz           # 537K sports posts with pre-computed candidate representations
  example_sequence.json       # Example user action history (3 posts: NFL, NBA, NHL)
```



#### 2. Run the pipeline


```shell
uv run run_pipeline.py --artifacts_dir artifacts/oss-phoenix-artifacts
```



This will:
1. Load the retrieval and ranking models from the exported checkpoints
2. Load the example user history (3 sports posts the user liked and dwelled on)
3. **Retrieve** the top-200 most relevant posts from the 537K sports corpus using dot-product similarity
4. **Rank** the retrieved posts by predicted engagement (favorite, reply, repost, dwell, video view)
5. Print the final ranked list with per-action engagement probabilities

#### 3. Customize

- **Change the user history**: Edit `example_sequence.json` to add your own post interactions. Each history item needs a `post_id`, `author_id`, and `actions` (action index to value mapping). Action indices follow the proto `ActionName` enum: `1` = favorite, `4` = reply, `5` = quote, `6` = repost, `11` = dwell, `13` = video quality view.
- **Adjust retrieval depth**: Use `--top_k_retrieval 500` to retrieve more candidates before ranking.
- **Display more results**: Use `--top_k_display 50` to show more ranked results.

### Model Architecture (Mini Config)

| Parameter | Value |
|---|---|
| Embedding dimension | 128 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Key size | 32 |
| Widening factor | 2 |
| History sequence length | 127 |
| Candidate sequence length | 64 |
| User/Item/Author vocab | 1,000,000 each |
| Hashes per entity | 2 |
| Action types | 19 |

### Running Tests


```shell
uv run pytest test_recsys_model.py test_recsys_retrieval_model.py
```
