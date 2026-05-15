# Thesis Knowledge Base - NGUYỄN THÀNH PHÁT

**Thesis:** Causal Discovery on Time Series Data  
**Topic:** ADIA Lab Causal Discovery Challenge - 8-class classification of causal roles  
**Language:** English (with Vietnamese bilingual front matter)  
**Status:** Partially written; needs Chapter 5 completion  

---

## OVERVIEW

This thesis documents participation in the **ADIA Lab Causal Discovery Challenge** (via CrunchDAO), a machine learning competition requiring classification of causal variable roles into 8 classes: Confounder, Mediator, Collider, Independent, Cause of X, Consequence of X, Cause of Y, Consequence of Y.

**Key Innovation:** Dual-pipeline neural architecture combining 1D edge tensor processing (kernel regression + ANM residuals) with 2D scatter density visual analysis, achieving **81.08% Balanced Accuracy** (vs. competition winner 76.70%).

---

## THESIS STRUCTURE

```
NGUYỄN_THÀNH_PHÁT___KLTN_T5_2026/
├── thesis.tex                     # Main document (175 lines)
├── cover.tex                      # Triple cover (VI/EN, TikZ borders)
├── references.bib                 # 268 lines, 32 citations
├── outline.md                     # Chapter guidance notes
├── figures/
│   ├── fig_edge_tensor.tex        # 8-channel tensor TikZ (243 lines)
│   ├── fig_arch_overview.tex      # Architecture diagram TikZ (186 lines)
│   ├── uet.jpg                    # UET logo
│   └── image.png
└── chapters/
    ├── acknowledgement.tex        # Lời cảm ơn
    ├── assurance.tex              # Lời cam đoan
    ├── abtract_vi.tex            # Tóm tắt tiếng Việt
    ├── abtract_en.tex            # Abstract English
    ├── glossary.tex               # Danh mục từ viết tắt
    ├── conclusion.tex             # Kết luận (needs expansion)
    ├── c1/c1_introduction.tex     # Chapter 1: Introduction (214 lines)
    ├── c2/c2_chapter.tex          # Chapter 2: Background (215 lines)
    ├── c2/c2_additions.tex        # Additional background
    ├── c3/c3_chapter.tex          # Chapter 3: Method (307 lines)
    ├── c4/c4_chapter.tex          # Chapter 4: Evaluation (194 lines)
    ├── c5/c5_chapter.tex          # Chapter 5: Results (239 lines)
    ├── c5/c5_classification.tex   # Classification results
    └── c5/c5_detection.tex        # Detection analysis
```

---

## KEY CHAPTERS TO COMPLETE

### outline.md Guidance:
- **Chapter 2:** All domain knowledge, problem description, existing solutions
- **Chapter 3:** Detailed solution description
- **Chapter 4:** Experimental methodology, evaluation approach
- **Chapter 5:** Update numerical results (needs latest figures)

---

## TECHNICAL CONTRIBUTIONS

### 1. Edge Tensor Extension (3 → 8 channels)
- 3 multi-bandwidth kernel channels (h ∈ {0.2, 0.5, 1.0})
- 3 multi-bandwidth ANM residual channels
- +1.48% (local) / +1.50% (public) accuracy gain

### 2. Structural Attention Bias
- Topology-aware attention (6 learnable scalars)
- Encodes chain/fork/collider relationships
- +2.65% accuracy gain

### 3. X/Y Remap Data Augmentation
- Treats each DAG edge as new (X', Y') pair
- Recomputes 8-class labels for all variables
- ~11.2× training data multiplication
- +1.55-3.32% accuracy gain

### 4. 2D Scatter Density Pipeline
- Visual representation of pairwise relationships
- CNN encoder parallel to edge tensor pipeline
- +0.88% accuracy gain

**Final Score:** v11 + XY aug → **81.08%** (public leaderboard)

---

## LATEX PACKAGES & CONVENTIONS

### Document Class
```latex
\documentclass[a4paper,13pt]{report}
\usepackage[vietnamese, main=english]{babel}
```

### TikZ Libraries
```latex
\usetikzlibrary{arrows.meta, positioning, calc}
```

### Key Packages
- `tikz` - Custom DAG and architecture diagrams
- `algorithm` - Pseudocode blocks
- `natbib[sort&compress]` - Citations (unsrt style)
- `listings` - Code highlighting with bg_gray (RGB 242,242,235)
- `booktabs` - Professional tables
- `threeparttable` - Notes under tables
- `subcaption` - Subfigures

### Custom Commands
- `\argmax` - Argmax operator
- `\cev{}` - Reflected vector arrow

---

## BIBLIOGRAPHY CATEGORIES

| Category | Key Citations |
|----------|---------------|
| **Foundations** | `pearl2009causality`, `spirtes2000causation`, `peters2017elements` |
| **Methods** | `shimizu2006lingam`, `hoyer2009anm`, `geiger2013dseparation` |
| **Competition** | `crunchdao_causal_2024`, `olivetti2025adialab` |
| **Solutions** | `adia_rank1_report`, `adia_rank4_medium` |
| **Deep Learning** | `vaswani2017attention`, `he2016resnet` |
| **Optimization** | `loshchilov2019adamw`, `lin2017focalloss` |

---

## FIGURES

### TikZ Diagrams
1. **fig_edge_tensor.tex** - 8-panel tensor visualization
   - Row 1: Sorted u, rank-sorted v
   - Row 2-3: 3 NW kernel bandwidths, 3 ANM residuals
   - Uses `\panelclip` macros

2. **fig_arch_overview.tex** - Dual-pipeline architecture
   - Left (blue): Edge Context Encoder
   - Right (teal): Node Visual Classifier
   - Center (orange): Fusion & Output

### Key Images
- `eight_roles_dag.png` - Illustration of 8 causal roles
- `uet.jpg` - University logo

---

## BUILD COMMANDS

```bash
# Navigate to thesis folder
cd /home/phatnguyen11/code/Graduation-Thesis-BsC-2026/NGUYỄN_THÀNH_PHÁT___KLTN_T5_2026/

# Quick build
latexmk -pdf thesis.tex

# Manual 4-pass build
pdflatex thesis.tex
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex

# Clean auxiliary files
latexmk -c
```

---

## ML CODEBASE (src/)

Companion implementation in `../src/`:
- `v11-v30.py` - Model version experiments
- `evaluate.py` - Evaluation utilities
- `*.ipynb` - Analysis notebooks

**Latest version:** v26e_xyaug.py (XY augmentation implementation)

---

## WHAT NEEDS WORK

1. **Chapter 5** - Update with latest experimental results
2. **Conclusion** - Expand limitations and future work sections
3. **Figures** - Add any missing TikZ diagrams for Chapter 4
4. **Citations** - Verify all referenced works are in .bib file
5. **Abstracts** - Ensure VI/EN abstracts match final content

---

## PEER REVIEW NOTES

When reviewing peer theses:
- Compare their approach to your dual-pipeline architecture
- Note their use of Vietnamese vs. English technical terms
- Observe figure conventions (Quân has 16 workflow diagrams)
- Note survey methodology (Minh has 48-educator survey)

---

## SUPERVISORS

- **TS. Nguyễn Văn Sơn** (Dr. Nguyen Van Son) - Primary
- Institution: Vietnam National University, Hanoi - UET
