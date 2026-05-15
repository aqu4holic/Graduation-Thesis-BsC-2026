# Chapter 2 Revision Plan
## Background and Related Work

**Status:** Plan Mode - Awaiting Approval Before Writing  
**Target Length:** ~3,000-3,200 words  
**Constraint:** NO Gap Analysis section (moved to C5 per user request)

---

## Executive Summary

Current C2 has strong foundations (causal graphs, 8-class problem, edge tensors, deep learning basics) but lacks background for two critical C3 contributions:
1. **Structural Attention Bias** - C3 uses topology-aware attention; C2 only covers vanilla self-attention
2. **X/Y Remap Augmentation** - C3 multiplies data 11×; C2 has no augmentation background

Additionally, c2_additions.tex (2D density, 2D ConvNets) needs integration, and label derivation logic needs to move from C4.

---

## Proposed Section Structure

### Section 2.1: Causal Graph Theory
**Status:** KEEP (minor edits)  
**Words:** ~450  
**Content:**
- DAGs and Structural Causal Models (Pearl 2009)
- d-Separation and three motifs (chain, fork, collider)
- Current text is solid, well-cited

**Action:** No major changes needed.

---

### Section 2.2: The Eight-Class Problem
**Status:** ENHANCE (add label derivation)  
**Words:** ~650 (was ~500, +150 for label table)  

**Subsections:**
- **2.2.1 ADIA Lab Challenge Setup** - Keep current
- **2.2.2 Label Derivation from Adjacency Matrix** ⭐ **NEW** (move from C4)
  - Table showing 8-class rules from 4 adjacency bits
  - Confounder: K→X=1, X→K=0, K→Y=1, Y→K=0
  - Mediator: K→X=0, X→K=1, K→Y=1, Y→K=0
  - etc.
  - **Why here:** C4 should focus on evaluation; problem setup belongs in Background

- **2.2.3 Class Definitions** - Keep current 8 definitions
- **2.2.4 Classification Challenges** - Strengthen Mediator explanation
  - Current: Mentions Mediator vs Confounder confusion
  - **Add:** Why Mediator is harder than Confounder in practice (C5 data: 56% vs 75%)

---

### Section 2.3: Data Representation
**Status:** MERGE + UNIFY 1D and 2D  
**Words:** ~900 (combine original 2.3 + c2_additions)  

**Subsections:**
- **2.3.1 Edge Tensor Construction (1D)** - Keep current
  - 8-channel structure
  - Multi-bandwidth kernel regression
  - ANM residuals
  - Equations for Nadaraya-Watson

- **2.3.2 From 1D Curves to 2D Densities** ⭐ **NEW TRANSITION**
  - Motivation: Edge tensors capture pairwise relationships
  - Limitation: Can't see joint distribution with X AND Y simultaneously
  - Introduce 2D scatter density as complementary view

- **2.3.3 2D Scatter Density Images** (from c2_additions.tex)
  - 2D histogram construction
  - Gaussian smoothing (dual-σ: 4.0 for raw, 2.0 for ANM)
  - Visual conditional independence test concept

**Integration:** c2_additions.tex Section 2.5 → merged here as unified data representation section (1D→2D progression)

---

### Section 2.4: Deep Learning Architectures
**Status:** ENHANCE (add structural attention background)  
**Words:** ~750 (was ~550, +200 for structural bias)  

**Subsections:**
- **2.4.1 Convolutional Networks** (merge 1D + 2D)
  - Conv1D for edge tensors (keep current)
  - Conv2D for scatter images (from c2_additions.tex Section 2.6)
  - Residual connections, GroupNorm, GELU

- **2.4.2 Self-Attention Mechanisms** - Keep current vanilla attention
  - Standard multi-head attention (Vaswani 2017)
  - Query-Key-Value formulation

- **2.4.3 Graph Attention and Relational Inductive Bias** ⭐ **NEW - CRITICAL**
  **Purpose:** Background for C3's structural attention bias (6 learnable scalars)
  
  **Content:**
  - Graph Attention Networks (GAT) [Veličković et al. 2018]
    - Attention over graph edges
    - Learning edge-specific weights
  - Relational inductive biases [Battaglia et al. 2018]
    - Encoding structure into neural networks
    - Why structure matters for reasoning
  - Attention bias mechanisms
    - Relative positional encodings
    - Learned vs handcrafted biases
  
  **Why critical:** Without this, C3's 6-scalar topology bias seems arbitrary. With this, it's a natural application of GAT principles.

---

### Section 2.5: Training Methodology
**Status:** ENHANCE (add augmentation background)  
**Words:** ~350 (was ~200, +150 for augmentation)  

**Subsections:**
- **2.5.1 Loss Functions and Optimization** - Keep current
  - Weighted cross-entropy
  - AdamW, cosine annealing

- **2.5.2 Data Augmentation for Graphs** ⭐ **NEW**
  **Purpose:** Background for C3's X/Y remap (~11× data multiplication)
  
  **Content:**
  - Data augmentation in structured domains
  - Label-preserving transformations
  - Graph symmetries and equivalent views
  - Multi-view learning principles
  
  **Why:** X/Y remap treats each DAG edge as new (X',Y') pair. Needs theoretical grounding.

---

### Section 2.6: Existing Approaches on ADIA Lab Challenge
**Status:** KEEP (restructure ending)  
**Words:** ~450  

**Subsections:**
- **2.6.1 First-Place Solution** - Keep current (76.70%, 188K params)
- **2.6.2 ML-based Approaches** - Keep current (feature engineering + boosting)
- **2.6.3 Position of This Thesis** - Revise
  - Remove gap analysis language (moved to C5)
  - Keep: Reimplementation baseline (v2: 73.96%)
  - Keep: Target improvements (Mediator ~56%, Collider ~63%)
  - Keep: Reference to Chapter 3 for contributions

**Deletion:** Remove c2_additions.tex "Gap Analysis" section (user confirmed this belongs in C5 Results)

---

## Word Count Summary

| Section | Current | Target | Δ |
|---------|---------|--------|---|
| 2.1 Causal Graph Theory | 450 | 450 | 0 |
| 2.2 Eight-Class Problem | 500 | 650 | +150 |
| 2.3 Data Representation | 550 | 900 | +350 |
| 2.4 Deep Learning | 550 | 750 | +200 |
| 2.5 Training | 200 | 350 | +150 |
| 2.6 Existing Approaches | 400 | 450 | +50 |
| **Total** | **~2,650** | **~3,100** | **+450** |

---

## Critical New Content Details

### 2.4.3 Graph Attention Background (200 words)

**Key citations to add:**
```bibtex
@inproceedings{velickovic2018gat,
  title={Graph Attention Networks},
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adri{\`{a}} and Li{\`{o}}, Pietro and Bengio, Yoshua},
  booktitle={ICLR},
  year={2018}
}

@article{battaglia2018relational,
  title={Relational inductive biases, deep learning, and graph networks},
  author={Battaglia, Peter W and Hamrick, Jessica B and Bapst, Victor and others},
  journal={arXiv preprint arXiv:1806.01261},
  year={2018}
}
```

**Content outline:**
- GAT extends self-attention to graphs by computing attention over edges
- Key innovation: edge-specific attention weights instead of uniform aggregation
- Relational inductive bias: baking structural assumptions into architecture
- Connection to thesis: topology-aware attention bias (6 scalars) is lightweight GAT

### 2.5.2 Data Augmentation for Graphs (150 words)

**Key citations to add:**
```bibtex
@inproceedings{feng2022graph,
  title={Graph Random Neural Networks for Semi-Supervised Learning on Graphs},
  author={Feng, Wenzheng and others},
  booktitle={NeurIPS},
  year={2022}
}

@article{Shorten2019DataAugmentation,
  title={A survey on Image Data Augmentation for Deep Learning},
  author={Shorten, Connor and Khoshgoftaar, Taghi M.},
  journal={Journal of Big Data},
  year={2019}
}
```

**Content outline:**
- Data augmentation increases effective training data without collection cost
- In graphs: label-preserving transformations exploit symmetries
- Multi-view learning: treating different perspectives of same structure as distinct samples
- Connection to thesis: X/Y remap exploits that any DAG edge can serve as treatment-outcome pair

---

## Integration of c2_additions.tex

| Source Section | Action | Target Location |
|----------------|--------|-----------------|
| 2.5 2D Density Estimation | Merge into | 2.3.3 |
| 2.6 2D Convolutional Networks | Merge into | 2.4.1 |
| 2.7 Gap Analysis | **DELETE** | N/A (moved to C5) |

After integration, c2_additions.tex can be deprecated or deleted.

---

## Citation Requirements

**Must add to references.bib:**
1. Veličković et al. 2018 (GAT)
2. Battaglia et al. 2018 (Relational inductive biases)
3. Feng et al. 2022 (Graph augmentation) or similar
4. Shorten & Khoshgoftaar 2019 (Data augmentation survey)

**Already present:**
- Pearl 2009, Spirtes 2000, Geiger 2013 (causal foundations)
- Vaswani 2017 (attention)
- He 2016 (ResNet)
- Wu 2018 (GroupNorm)
- Loshchilov 2019 (AdamW)

---

## Files to Modify

1. **chapters/c2/c2_chapter.tex** - Main revisions
2. **chapters/c4/c4_chapter.tex** - Remove label derivation table (move to C2)
3. **references.bib** - Add 4 new citations
4. **chapters/c2/c2_additions.tex** - Deprecated after merge (can delete)

---

## Approval Checklist

Before I start writing, please confirm:

- [ ] **Remove Gap Analysis from C2** - Move to C5 instead
- [ ] **Add Graph Attention background (2.4.3)** - ~200 words on GAT + relational bias
- [ ] **Add Data Augmentation background (2.5.2)** - ~150 words on graph augmentation
- [ ] **Merge c2_additions.tex content** - 2D density into 2.3, 2D Conv into 2.4
- [ ] **Move label derivation from C4 to C2** - Table showing adjacency → class mapping
- [ ] **Target length ~3,100 words** - Up from current ~2,650

**Reply with "approve" or specific changes, then I'll begin drafting.**
