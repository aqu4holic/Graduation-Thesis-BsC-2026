# Chapter 1: Introduction - Revised Plan v2

## Chapter Purpose
Set up the research problem with compelling persuasion: existing methods have clear **limitations**, our solution overcomes them, achieving SOTA (81.082%). Guide the reader from context to problem to solution to contributions.

## Writing Rules Checklist
- [x] No `[]` in figure commands
- [x] All figures have interpretative descriptions (introduce → then describe in detail)
- [x] No parentheses in titles
- [x] No 1-2 sentence paragraphs (except section openings)
- [x] No sentences >3-4 lines
- [x] Use `` for quotes in Overleaf
- [x] No widow lines
- [x] Limitations **bolded** for easy scanning

## Section Structure

### §1.1 Opening: Context and Problem Statement
**Purpose:** Establish why causal discovery matters

**Content:**
- Opening hook: Correlation vs causation - why distinguishing cause from effect matters in real decisions
- Problem: ADIA Lab Causal Discovery Challenge - classify 8 causal roles
- Eight roles: Confounder, Mediator, Collider, Independent, Cause of X/Y, Consequence of X/Y
- Challenge: subtle class distinctions, class imbalance, permutation invariance, nonlinear relationships

**No citations here** - this is general context

**Paragraph flow:**
- Paragraph 1: Real-world impact (1-2 sentences, punchy)
- Paragraph 2: The problem defined (1-2 sentences)
- Paragraph 3: Eight roles briefly listed (2-3 sentences)
- Paragraph 4: Four challenges introduced (3-4 sentences max each)

---

### §1.2 Existing Solutions and Their Limitations
**Purpose:** Grouped review of prior work with **bolded limitations** that our method overcomes

**Structure:** Group solutions, then explicitly state limitations in **bold**

**Group 1: Classical Causal Discovery Methods**
- PC algorithm, conditional independence tests
- **Limitation 1: Limited to linear relationships and simple dependencies**
- **Limitation 2: Cannot handle the 8-class fine-grained classification task**
- Score: ~40% Balanced Accuracy

**Group 2: Machine Learning Feature Engineering**
- v10: LightGBM on scalar features (63.11%)
- v13: Full ML pipeline with 300+ features, ensemble methods (72.64%)
- **Limitation 3: Scalar features irreversibly discard curve shape information**
- **Limitation 4: 300+ features but still below DL baseline - compression is the bottleneck**
- **Limitation 5: Manual feature engineering doesn't scale to complex causal structures**

**Group 3: Deep Learning Approaches**
- Competition winner (Rank 1): 3-channel edge tensors + attention (76.70%)
- **Limitation 6: Only 3 channels - misses multi-scale patterns**
- **Limitation 7: No structural awareness in attention mechanism**
- **Limitation 8: No 2D visual representation of joint distributions**

**Persuasion paragraph:** After listing limitations, write 1-2 sentences explaining why these limitations make our approach necessary

---

### §1.3 Our Solution and Results
**Purpose:** Present our method and headline results

**Content:**
- Dual-pipeline architecture: 1D edge context + 2D scatter density
- Five contributions summarized (brief, not detailed - details in later sections)
- Headline result: 81.082% on private test set
- Comparison: +4.38% over winner, +7.12% over baseline

**3-4 sentences max** - this is a transition paragraph

---

### §1.4 Five Technical Contributions
**Purpose:** Detailed contributions with specific gains

**Structure:** Each contribution as a flowing paragraph (not bullet list)

**Contribution 1: 8-Channel Edge Tensor**
Extends 3-channel baseline to 8 channels via multi-bandwidth kernel regression and ANM residuals. This captures causal asymmetry at multiple scales. Gain: +1.89%.

**Contribution 2: Structural Attention Bias**
Injects topology-aware priors into attention layers with only 6 learnable scalars. This encodes chain, fork, collider relationships directly. Gain: +2.65%.

**Contribution 3: 2D Scatter Density Pipeline**
Complements 1D edge tensors with visual representations of joint distributions. This captures patterns invisible to pairwise analysis. Gain: +0.88%.

**Contribution 4: Dual-Head Loss Weighting**
Uses lambda=0.7 weighting between node and edge loss heads. This ensures edge embeddings remain causally meaningful throughout training. Gain: +0.59%.

**Contribution 5: X/Y Remap Augmentation**
Treats each DAG edge as a new treatment-outcome pair. This multiplies training data by ~11x without collection. Gain: +1.55%.

---

### §1.5 Validation Evidence
**Purpose:** Ablation proof that contributions are real

**Content:**
- Removing edge context: 80.47% → 51.51% (-29 points)
- Structural bias outperforms thousands of unconstrained parameters
- Each ablation validates one contribution

**2-3 paragraphs**

---

### §1.6 Thesis Organization
**Purpose:** Roadmap for remaining chapters

**Content:**
- Chapter 2: Background (causal graphs, edge tensors, deep learning components)
- Chapter 3: Proposed Method (full pipeline)
- Chapter 4: Evaluation (dataset, metrics, baselines)
- Chapter 5: Results (ablation, analysis, lessons)

**Keep brief - 1 paragraph**

---

## Figure Requirements
- [ ] **Table 1.1: Performance comparison** - All methods with scores and limitations listed
  - Must have interpretative caption describing what to look for
  - No `[]` options in table environment

---

## Citation Requirements
- Pearl 2009 (causal foundations)
- Olivetti 2025 (competition analysis)
- Rank 1 solution report
- v13 ML fullstack paper if available

## Current Status
⏳ PENDING REVISION - Need to rewrite following this structure

## Key Changes from Previous Version
1. Opening now has compelling hook about real-world impact
2. Limitations grouped and **bolded** for easy scanning
3. Persuasion paragraph after limitations
4. Contributions written as paragraphs, not bullets
5. No orphaned 1-2 sentence sections
6. Sentence length monitored (3-4 lines max)