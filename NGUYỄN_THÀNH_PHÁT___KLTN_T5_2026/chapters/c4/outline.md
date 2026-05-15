# Chapter 4: Evaluation Methodology - Revised Plan v2

## Chapter Purpose
Describe dataset, evaluation metric, experimental baselines, and procedures. This chapter is factual - present information clearly without persuasion.

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

### §4.1 Dataset
**Purpose:** Characterize the data used

**Subsections:**

**§4.1.1 Data Source**
- ADIA Lab Causal Discovery Challenge via CrunchDAO
- Training: 23,500 instances
- Public test: 1,880 instances
- Private test: held-out partition
- Final score: 81.082% (private)

**§4.1.2 Label Derivation from Adjacency Matrix**
- Table 4.1: Derivation rules (4 binary bits → 8 roles)
- Each role defined by its 4-bit pattern
- All variables normalized to [-1,1]

**§4.1.3 Dataset Characteristics**
- n=1000 observations per instance
- p∈{3,...,10} variables
- Class imbalance: Independent most frequent, Mediator/Collider rarest
- Generation: neural networks, Gaussian processes, Bayesian models
- 6 synthetic graph types + food-web graphs

---

### §4.2 Evaluation Metric: Balanced Accuracy
**Purpose:** Define and justify the metric

**Definition:**
BA = (1/C) Σ_c TP_c / (TP_c + FN_c) for C=8 classes

**Rationale (3 paragraphs):**
1. Class imbalance requires per-class recall averaging
2. All 8 roles equally important in applications
3. baselines: random=12.5%, organizer baseline~58%, winner=76.70%

---

### §4.3 Experimental Baselines
**Purpose:** Establish comparison points

**Subsections:**

**§4.3.1 External Baselines**
- **Competition first-place (Rank 1)**: 76.70%, architecture description
- **Reimplementation (v2)**: 73.96% local / 73.61% public
- **ML baselines**: v10 (63.11%), v13 (72.64%)

**§4.3.2 Internal Ablation Baselines**
- v2: 3ch baseline (73.96%)
- v8b: 8ch 1D (76.94%)
- v11: 8ch + structural bias (79.59%)
- v11+xyaug: previous best (81.14%)
- v26c: 2D only (51.51%)
- v26e base: 8ch 2D + edge (80.47%)

---

### §4.4 Experimental Setup
**Purpose:** Hardware and procedures

**Subsections:**

**§4.4.1 Hardware Environment**
- 2× NVIDIA RTX 5880 GPUs (48GB)
- 48-core CPU, 251GB RAM
- DDP training

**§4.4.2 Two-Tier Evaluation**
- Local test: 1,880 instances with known ground truth
- Public LB: selected submissions
- Private test: final submission only (81.082%)

**§4.4.3 Experiment Design Principles**
- Ablation-first protocol
- Local validation before LB submission
- Avoids overfitting to public test

---

### §4.5 Complete Experiment Summary
**Purpose:** All versions in chronological table

**Table 4.2: All Results**
- Grouped by category (ML, DL, feature engineering, structural bias, 2D pipeline, final)
- Columns: Version, Description, Local, LB, Δv2
- Comprehensive but clean presentation

**Figure 4.1: Ablation Progression** (optional)
- If included: must have interpretative caption
- Shows cumulative gains visually

---

### §4.6 Chapter Summary
**Purpose:** Recap evaluation framework (brief)

**Content:**
- Dataset: 23,500 train + 1,880 local test
- Metric: Balanced Accuracy
- Baselines: ML (v10: 63.11%, v13: 72.64%), DL (v2: 73.96%)
- Final model: 81.082%

---

## Table Checklist
- [ ] Table 4.1: Label derivation rules
- [ ] Table 4.2: All experiment results

All tables must have:
- Centering
- Consistent caption formatting (caption above table)
- Interpretative captions (introduce → describe)

## Citation Requirements
- CrunchDAO ADIA docs
- Olivetti 2025 (post-competition analysis)
- Rank 1-5 solution reports

## Current Status
✅ STRUCTURALLY COMPLETE - Verify tables have proper formatting and captions