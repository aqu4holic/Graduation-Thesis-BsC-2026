# Chapter 5: Results and Analysis - Revised Plan v2

## Chapter Purpose
Present experimental results with mechanistic analysis. Show how each contribution improves performance and synthesize general lessons. This chapter builds the evidence that our approach works.

## Writing Rules Checklist
- [x] No `[]` in figure commands
- [x] All figures have interpretative descriptions
- [x] No parentheses in titles
- [x] No 1-2 sentence paragraphs
- [x] No sentences >3-4 lines
- [x] Use `` for quotes in Overleaf
- [x] No widow lines
- [x] Tables centered with consistent caption formatting

## Section Structure

### §5.1 Deep Learning vs Machine Learning
**Purpose:** Establish mechanistic why DL outperforms ML

**Subsections:**

**§5.1.1 v10: LightGBM Baseline (63.11%)**
- Scalar features only
- Encodes dependence strength, not curve shape
- Paragraph explaining the limitation

**§5.1.2 v13: Full ML Fullstack (72.64%)**
- 300+ features, 4 algorithms, ensemble
- Maximum effort under ML paradigm
- Still below DL baseline

**§5.1.3 Why Scalar Compression Fails**
- Key insight paragraph: irreversible information loss
- Edge tensor retains full n-length curves
- Conv1D learns which curve aspects discriminate

**Table 5.1: DL vs ML Comparison**
- v10, v13, proposed model
- Advantages and limitations columns

---

### §5.2 Feature Engineering Contributions
**Purpose:** Analyze channel extension gains

**Subsections:**

**§5.2.1 Multi-Bandwidth Kernel Regression (+1.48%)**
- v5m vs v2 comparison
- Three bandwidths capture different scales
- Public LB confirmation (+1.97%)

**§5.2.2 ANM Residual Channels (+1.50%)**
- v8b vs v5m
- Improvement concentrated in Collider (+6.4%), Consequence of X (+5.0%)
- Residual asymmetry encodes direction

**§5.2.3 X/Y Augmentation (+3.32% on v8b, +1.55% on v11)**
- v8b+xyaug vs v8b
- v11+xyaug vs v11
- Smaller gain on stronger base (structural bias already encodes X/Y relationships)
- Local-to-LB gap ≤0.35% confirms no overfitting

---

### §5.3 Structural Attention Bias
**Purpose:** Analyze the most impactful architectural change

**Subsections:**

**§5.3.1 Result and Mechanism (+2.65%)**
- v11 vs v8b
- Largest single gain
- Distributed across Confounder (+4.46%), Collider (+3.40%), Mediator (+1.04%)

**§5.3.2 Why Structural Bias Works**
- Contrast with failed experiments (v3, v4, v6, v9)
- Key principle: structured inductive bias > unconstrained capacity
- 6 scalars with alignment outperforms thousands of parameters without

---

### §5.4 2D Scatter Density Pipeline
**Purpose:** Analyze visual representation contribution

**Subsections:**

**§5.4.1 Edge Context Necessity**
- Table 5.2: Ablation without edge pipeline
- v26c: 51.51% (-29 points)
- v28b: 46.39%
- Edge context is structural necessity, not auxiliary

**§5.4.2 1D vs 2D Representation (+0.88%)**
- Table 5.3: Per-class breakdown
- 1D stronger for Collider
- 2D stronger for Confounder, Mediator, Consequence of Y
- Complementary, neither dominates

**§5.4.3 Channel Count: 12ch vs 8ch**
- Kernel coefficient maps inappropriate for Conv2D spatial bias
- 8-channel configuration is correct

**§5.4.4 Loss Lambda Tuning**
- Table 5.4: λ ablation
- λ=0.7 empirically optimal
- Training dynamics explanation

**§5.4.5 Final Model Results**
- Table 5.5: Per-class recall
- Local 81.19%, LB 80.96%, Private 81.082%
- Hardest: Confounder (74.93%), Mediator (73.71%), Collider (78.70%)

---

### §5.5 Full Ablation Table
**Purpose:** Cumulative contribution visualization

**Table 5.6: Incremental Ablation**
- All versions with gains
- Shows stepwise improvement from v2 to final

**Figure 5.1: Ablation Visualization** (optional)
- If included: must have interpretative caption

---

### §5.6 Synthesis and Lessons Learned
**Purpose:** Extract general principles

**Subsections:**

**§5.6.1 Shape Information Cannot Be Recovered**
- v13 ML fullstack: 72.64%
- All Conv1D models with curve channels: higher
- Implication for future causal discovery systems

**§5.6.2 Structured Inductive Bias Outperforms Unconstrained Capacity**
- Structural attention bias: only successful architectural modification
- Failed: parallel branches, dual-path transformers, node-centric attention, deep sets
- 6 scalars > thousands of unconstrained parameters

**§5.6.3 1D Curves and 2D Density Images Are Complementary**
- 1D: directional pairwise
- 2D: joint patterns with X and Y simultaneously
- Fusion outperforms either alone

**§5.6.4 Edge Context Is the Foundation**
- 29-point collapse without edge pipeline
- Binary edge loss is essential supervision

---

### §5.7 Chapter Summary
**Purpose:** Recap key findings (brief)

**Content:**
- Six key findings summarized
- Final achievement: 81.082% (+4.38% over winner, +7.12% over baseline)

---

## Table Checklist
- [ ] Table 5.1: DL vs ML comparison
- [ ] Table 5.2: Edge context necessity
- [ ] Table 5.3: 1D vs 2D per-class breakdown
- [ ] Table 5.4: Lambda ablation
- [ ] Table 5.5: Final model per-class
- [ ] Table 5.6: Incremental ablation

All tables:
- Centering
- Caption above
- Interpretative caption (introduce → describe)

## Key Insights for Defense
1. Why deep learning wins: curve shape preserved vs scalar compression
2. Why structural bias works: domain knowledge encoded
3. Why 2D helps: joint patterns invisible in 1D
4. Why edges are essential: graph-wide context
5. Why augmentation helps: 11× data multiplication

## Current Status
✅ STRUCTURALLY COMPLETE - Verify all tables have proper formatting and captions