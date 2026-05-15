# Chapter 3: Proposed Method - Revised Plan v2

## Chapter Purpose
Present the complete dual-pipeline architecture with five technical contributions. This is the solution chapter - be precise, use equations, cite appropriately.

## Writing Rules Checklist
- [x] No `[]` in figure commands
- [x] All figures have interpretative descriptions (introduce what figure shows, then describe)
- [x] No parentheses in titles
- [x] No 1-2 sentence paragraphs
- [x] No sentences >3-4 lines
- [x] Use `` for quotes in Overleaf
- [x] No widow lines
- [x] Chapter title should be creative (see options below)

## Chapter Title Options
1. **"Neural Edge-Node Fusion: A Dual-Pipeline Architecture for Causal Discovery"**
2. **"Shape-Preserving Causal Classification via Multi-Scale Edge Tensors and Structural Attention"**
3. **"Beyond Correlation: Dual-Pipeline Learning for Eight-Class Causal Role Classification"**

Recommendation: Option 1 - specific and descriptive

## Section Structure

### §3.1 Architecture Overview
**Purpose:** High-level view of the dual-pipeline system

**Content:**
- Edge Context Encoder: processes all p(p-1) directed pairs
- Node Visual Classifier: processes each node K individually
- Fusion: combines 4 edge embeddings + 1 visual embedding → 8-class prediction
- Key insight: two parallel streams on different representations

**Figure 3.1: Architecture Overview**
- Full pipeline diagram
- Interpretative caption: describes the flow from input data through both pipelines to fusion

---

### §3.2 Preprocessing and Edge Tensor Construction (Contribution 1)
**Purpose:** Data preparation and the 8-channel representation

**Subsections:**

**§3.2.1 Data Normalization**
- Z-score standardization equation
- Scale-invariance for kernel bandwidth

**§3.2.2 Complete Directed Graph**
- p(p-1) ordered pairs
- Directional asymmetry preserved

**§3.2.3 8-Channel Edge Tensor**
- Channels 1-2: Sorted observations (paragraph explaining each)
- Channels 3-5: Multi-bandwidth kernel regression equations
- Channels 6-8: ANM residual equations
- Gain statement: +1.89% over baseline

**Figure 3.2: Edge Tensor Visualization**
- 8 channels shown as 1D signals
- Interpretative caption: explains the three rows (observations, kernel regression, ANM residuals)

---

### §3.3 Edge Context Encoder
**Purpose:** 1D pipeline for edge tensor processing

**Subsections:**

**§3.3.1 Per-Edge Feature Extraction**
- Stem layer: 8→64 channels
- 5 Residual Conv1D blocks
- Global Average Pooling

**§3.3.2 Edge Type Embedding**
- 7-class taxonomy
- Learnable embedding table
- Merge operation

**§3.3.3 Binary Edge Head**
- Auxiliary supervision only
- Forces causally meaningful embeddings

**Equations:** All displayed properly (no `[]` in align environment)

---

### §3.4 Structural Attention Bias (Contribution 2)
**Purpose:** Topology-aware attention mechanism

**Subsections:**

**§3.4.1 Motivation**
- Standard attention ignores topology
- Edges sharing endpoints should attend more

**§3.4.2 Six Topological Relationship Types**
- Table 3.1: Types, conditions, interpretations
- Reverse, shared source, shared target, forward chain, backward chain, unrelated

**§3.4.3 Learned Scalar Bias Implementation**
- Attention logit modification equation
- 6 additional parameters per head
- Initialization to zero

**§3.4.4 Why This Succeeds**
- Contrast with failed experiments
- Structured inductive bias vs unconstrained capacity
- Gain: +2.65%

**Figure 3.3: Topological Relationship Types**
- Diagram showing the 6 types
- Interpretative caption: explains each type with causal interpretation

---

### §3.5 Node Visual Representation (Contribution 3)
**Purpose:** 2D pipeline for complementary visual features

**Subsections:**

**§3.5.1 Motivation**
- 1D edge tensor is pairwise
- Missing: joint visual pattern of K with X AND Y

**§3.5.2 8-Channel Scatter Density Image**
- Channels 1-4: Raw scatter density (σ_raw=4.0)
- Channels 5-8: ANM residual scatter (σ_ANM=2.0)
- Resolution: 32×32

**§3.5.3 Hierarchical Conv2D Feature Extractor**
- Progressive downsampling: 32→16→8→4
- AdaptiveAvgPool to 64-dim

**Figure 3.4: Node Image Construction**
- Shows how scatter plot becomes density image
- Interpretative caption: step-by-step transformation explained

---

### §3.6 Fusion and Node Classification
**Purpose:** Combine edge context with visual features

**Subsections:**

**§3.6.1 Combining Representations**
- 5 sources: 4 edge embeddings + 1 visual embedding
- Learned merge operator
- Node head

**§3.6.2 Necessity of Edge Context**
- Ablation table showing collapse without edge pipeline
- v26c: 51.51% (-29 points)
- Edge context is structural necessity

**Equations:** Node merge, node head

---

### §3.7 Loss Function and Training Objective (Contribution 4)
**Purpose:** Dual-head training with optimal weighting

**Subsections:**

**§3.7.1 Dual-Head Loss with Lambda Weighting**
- Node loss: weighted cross-entropy
- Edge loss: binary cross-entropy
- Total loss equation

**§3.7.2 Rationale for λ=0.7**
- Counterintuitive: edge head discarded at inference
- Mechanism explanation
- Ablation comparison

**Equation:** Total loss with λ

---

### §3.8 X/Y Remap Data Augmentation (Contribution 5)
**Purpose:** Multiply training data without collection

**Subsections:**

**§3.8.1 Strategy**
- Each edge becomes new (X',Y') pair
- Label recomputation for all nodes
- ~11× multiplication

**§3.8.2 Efficient Implementation**
- Kernel regression computed once per graph
- Fast per-remap operations

**§3.8.3 Gains Across Configurations**
- v8b: +3.32%
- v11: +1.55%
- v26e: +0.72%

---

### §3.9 Training Configuration
**Purpose:** Fixed hyperparameters

**Content:**
- AdamW, cosine annealing, LR=10^-3
- Batch size 64, 10 epochs on augmented data
- 2× RTX 5880 GPUs, DDP
- ~280K parameters

---

### §3.10 Chapter Summary
**Purpose:** Recap five contributions and final result

**Content:**
- Summary paragraph for each contribution
- Final score: 81.082%

---

## Figure Checklist
- [ ] Figure 3.1: Architecture overview (full pipeline)
- [ ] Figure 3.2: Edge tensor (8 channels)
- [ ] Figure 3.3: Topological relationship types
- [ ] Figure 3.4: Node image construction

All figures must have interpretative captions (introduce → describe in detail)

## Ablation Chain
v2 (73.96%) → v5m (75.44%) → v8b (76.94%) → v11 (79.59%) → v11+xyaug (81.14%) → v26e (80.47%) → v26e+xyaug (81.19%)

## Current Status
✅ STRUCTURALLY COMPLETE - Need to verify all figures have proper captions and equations are clean