# Chapter 2: Background and Related Work - Revised Plan v2

## Chapter Purpose
Provide theoretical foundations (heavy citations), define the 8-class problem precisely, and survey only **direct competitors** (related work goes in Introduction). This chapter should be densely cited.

## Writing Rules Checklist
- [x] No `[]` in figure commands
- [x] All figures have interpretative descriptions
- [x] No parentheses in titles
- [x] No 1-2 sentence paragraphs
- [x] No sentences >3-4 lines
- [x] Use `` for quotes in Overleaf
- [x] No widow lines
- [x] Heavy citation throughout (this chapter must cite extensively)

## Section Structure

### §2.1 Causal Graph Theory and the Eight Variable Roles
**Purpose:** Define problem formally with heavy citations

**Subsections:**

**§2.1.1 Structural Causal Models and DAGs**
- DAG definition with citations
- Structural equations
- The 8 roles derived from adjacency matrix (Table 2.1 with derivation rules)
- Citations: Pearl 2009, Spirtes 2000

**§2.1.2 The Eight Causal Roles**
- All 8 roles defined in flowing paragraphs (not enumerated list)
- Confounder: K→X, K→Y - common cause structure
- Mediator: X→K→Y - transmission pathway
- Collider: X→K←Y - induced dependence under conditioning
- Independent, Cause of X/Y, Consequence of X/Y - remaining structures
- Use citations for each definition
- Paragraph for each role (no bullets)

**§2.1.3 Challenges in Classifying Causal Roles**
- Confounder vs Cause of X: both induce K-X correlation
- Mediator vs Consequence of X: both have X→K
- Mediator vs Confounder: both produce X⊥Y|K but through different mechanisms
- Competition findings on hardest classes (cite Olivetti 2025)
- Citations throughout

---

### §2.2 Edge Tensor Representation
**Purpose:** Explain core data representation with citations

**Subsections:**

**§2.2.1 Motivation for Edge-Based Processing**
- Permutation equivariance
- Directional asymmetry encoding
- Citation: Rank 1 solution

**§2.2.2 The Eight Channels**
- Channels 1-2: Sorted observations
- Channels 3-5: Multi-bandwidth kernel regression (cite Nadaraya 1964, Watson 1964)
- Channels 6-8: ANM residuals (cite Shimizu 2006, Hoyer 2009)
- Each channel explained in paragraph form

**§2.2.3 From 1D Curves to 2D Densities**
- Joint distribution representation
- Gaussian smoothing
- Visual independence test interpretation
- Citations for methodology

---

### §2.3 Deep Learning Components
**Purpose:** Background on neural network components with citations

**Subsections:**

**§2.3.1 Convolutional and Residual Networks**
- Conv1D operation
- Residual connections (cite He 2016)
- Group Normalization (cite Wu 2018)
- GELU activation (cite Hendrycks 2016)

**§2.3.2 Self-Attention and Multi-Head Attention**
- Attention mechanism (cite Vaswani 2017)
- Graph Attention Networks (cite Velickovic 2018)
- Relational inductive biases (cite Battaglia 2018)

**§2.3.3 Loss Functions and Training**
- Dual-head architecture
- Weighted loss
- Citations

---

### §2.4 Existing Approaches on the ADIA Lab Challenge
**Purpose:** Survey direct competitors ONLY (related competitors in Introduction)

**Note:** This section should cite heavily - these are published solutions

**Subsections:**

**§2.4.1 First-Place Solution (Rank 1)**
- Architecture description
- Performance (76.70%)
- Citations: Rank 1 report, Olivetti 2025

**§2.4.2 Rank 4-5 Solutions: Feature Engineering**
- Statistical features + gradient boosting
- Multiple algorithm outputs (PC, LiNGAM, NOTEARS, GES)
- Citations: Rank 4/5 Medium posts

**§2.4.3 Position of This Thesis**
- Reimplements and extends Rank 1
- Baseline: 73.96% / 73.61%
- Five contributions that address the gaps

---

### §2.5 Data Augmentation for Causal Discovery
**Purpose:** Background on augmentation strategies

**Subsections:**

**§2.5.1 Label-Preserving Transformations**
- DAG symmetries
- Multi-view learning (cite Sun 2019)
- Edge relabeling strategies

**§2.5.2 X/Y Remap as Data Multiplication**
- Theory: each edge can be (X',Y')
- Label recomputation
- Approximate 11x multiplication

---

## Figure Requirements
- [ ] **Figure 2.1: Eight roles with DAG structures**
  - Must have interpretative caption: first explain what the figure shows, then describe key patterns
  - Note pairs with similar signatures visually marked

- [ ] **Figure 2.2: Edge tensor visualization**
  - 8 channels illustrated
  - Interpretative caption explaining each row

- [ ] **Figure 2.3: 2D density construction**
  - Scatter plot to histogram to smoothed density
  - Interpretative caption

---

## Citation Density Target
This chapter should have the highest citation density:
- Pearl 2009 (causality foundations)
- Spirtes 2000 (causal inference)
- Peters 2017 (elements of causal inference)
- Nadaraya 1964, Watson 1964 (kernel regression)
- Shimizu 2006 (LiNGAM)
- Hoyer 2009 (ANM)
- He 2016 (ResNet)
- Wu 2018 (GroupNorm)
- Hendrycks 2016 (GELU)
- Vaswani 2017 (Attention)
- Velickovic 2018 (GAT)
- Battaglia 2018 (relational)
- Loshchilov 2019, 2017 (AdamW, SGDR)
- Sun 2019 (multiview)
- Olivetti 2025 (competition analysis)
- Rank 1-5 reports/solutions

## What NOT to Include
- Related competitors (these go in Introduction)
- Extensive d-separation theory (mention only what's needed for the 8-role definitions)
- Unused deep learning components

## Current Status
⏳ PENDING REVISION - Need to restructure following this plan

## Key Changes from Previous Version
1. Removed "Gap Analysis" section (goes to C5)
2. Removed d-separation subsection (not needed - user said no d-sep)
3. Related work now only direct competitors
4. Paragraph format throughout (no enumerated lists for role definitions)
5. Heavy citation throughout
6. Removed stray subsection issue (was causing compile problems)