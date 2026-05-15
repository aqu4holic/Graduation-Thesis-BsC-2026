# Thesis Knowledge Base - BSc 2026

**Project:** Graduation Thesis Repository (Bachelor of Science 2026)
**Generated:** 2026-05-06
**Location:** `/home/phatnguyen11/code/Graduation-Thesis-BsC-2026/`

---

## OVERVIEW

This repository contains 3 BSc graduation theses from the class of 2026. Two theses are peer-reviewed, and one is the primary thesis being improved.

| Thesis | Author | Topic | Language | Status |
|--------|--------|-------|----------|--------|
| **PRIMARY** (Improve) | Nguyễn Thanh Phát | Causal Discovery on Time Series Data | English | Partial, needs completion |
| PEER 1 (Review) | Nguyễn Hoàng Quân | AI Test Generation (Qodo Plus) | Vietnamese | Template-based, modular |
| PEER 2 (Review) | Lê Hoàng Minh | Edlora - AI Teaching Platform | English | Complete with survey data |

---

## REPOSITORY STRUCTURE

```
.
├── NGUYỄN_THÀNH_PHÁT___KLTN_T5_2026/     # [PRIMARY] Causal Discovery Thesis
│   ├── thesis.tex                        # Main document
│   ├── chapters/c1-c5/                   # Chapter content
│   ├── figures/                          # TikZ and image figures
│   ├── references.bib                    # 268 citations (Pearl, Spirtes, etc.)
│   └── outline.md                        # Chapter notes for completion
├── Nguyễn_Hoàng_Quân___KLTN_T5_2026/     # [PEER] AI Test Generation
│   ├── thesis.tex
│   ├── chapters/c1-c5/                   # Highly modular structure
│   │   └── c1/ (context, problem, purpose)
│   │   └── c4/ (dataset, script, results, analysis)
│   ├── figures/                          # 16 workflow/technical diagrams
│   └── references.bib                    # 673 citations (LLM, self-healing)
├── LÊ_HOÀNG_MINH___KLTN_T5_2026/         # [PEER] Edlora Platform
│   ├── thesis.tex
│   ├── chapters/c1-c4/                   # Clean 4-chapter structure
│   ├── figures/                          # Survey charts, positioning charts
│   ├── references.bib                    # 183 citations (education, AI)
│   └── thesis.pdf                        # Compiled output exists
└── src/                                  # ML codebase (Phat's)
    ├── v*.py                             # Model experiments (v11-v30)
    ├── *.ipynb                           # Jupyter notebooks
    └── evaluate.py                       # Evaluation utilities
```

---

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| **Improve thesis** | `NGUYỄN_THÀNH_PHÁT___KLTN_T5_2026/` | Check `outline.md` for chapter guidance |
| **Peer review Quân** | `Nguyễn_Hoàng_Quân___KLTN_T5_2026/` | Modular structure with sub-files |
| **Peer review Minh** | `LÊ_HOÀNG_MINH___KLTN_T5_2026/` | Survey data + complete PDF |
| **Build LaTeX** | Any `thesis.tex` | Use `pdflatex` + `bibtex` |
| **Add citations** | `references.bib` | Each folder has own bibliography |

---

## THESIS-SPECIFIC PATTERNS

### NGUYỄN THÀNH PHÁT (Primary)
- **Topic:** ADIA Lab Causal Discovery Challenge - classifying 8 causal roles
- **Approach:** Neural network with edge tensors, kernel regression
- **Key figures:** `eight_roles_dag.png`, TikZ DAGs
- **Packages:** `tikz` (arrows.meta), `algorithm`, `natbib`
- **Status:** c1-c5 exist; needs Chapter 5 completion; has `outline.md`

### NGUYỄN HOÀNG QUÂN (Peer 1)
- **Topic:** Qodo Plus - Self-healing automated test generation
- **Approach:** LLM-based code generation with error recovery
- **Language:** Vietnamese with English technical terms
- **Key figures:** 16 diagrams (MAPE-K, workflow, LLM agent)
- **Structure:** Highly modular - c1/c4 split into sub-files
- **Citations:** 673 refs (Vaswani attention, self-healing, coverage)

### LÊ HOÀNG MINH (Peer 2)
- **Topic:** Edlora - Agentic AI for teaching slide development
- **Approach:** RAG + storytelling + gamification for education
- **Key figures:** Survey charts (48 educators), positioning chart
- **Structure:** Clean 4-chapter organization
- **Citations:** 183 refs (WEF jobs, gamification, storytelling)

---

## LATEX CONVENTIONS

### Document Class
All theses use `report` class with Vietnamese/English bilingual setup:
```latex
\documentclass[a4paper,13pt]{report}
\usepackage[utf8]{vietnam}
\usepackage[vietnamese, main=english]{babel}  # or main=english
```

### Chapter Organization
- **Phat:** c1-c5 (Introduction, Background, Method, Experiments, Results)
- **Quân:** c1-c5 (modular sub-files per chapter)
- **Minh:** c1-c4 (Introduction, Background, Method, Results)

### Common Packages
```latex
\usepackage{tikz}          # DAGs and figures
\usepackage{algorithm}     # Pseudocode
\usepackage{natbib}        # Citations (unsrt style)
\usepackage{booktabs}      # Tables
\usepackage{hyperref}      # Links
\usepackage{listings}      # Code blocks
```

### Citation Style
- Uses `natbib` with `unsrt` bibliography style
- Pattern: `\cite{authorYearKeyword}`
- Examples: `\cite{pearl2009causality}`, `\cite{vaswani2023attentionneed}`

---

## BUILD COMMANDS

```bash
# Navigate to thesis folder
cd NGUYỄN_THÀNH_PHÁT___KLTN_T5_2026/

# Build with latexmk (recommended)
latexmk -pdf thesis.tex

# Manual build
pdflatex thesis.tex
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex

# Clean auxiliary files
latexmk -c
```

---

## PEER REVIEW WORKFLOW

When conducting peer reviews:

1. **Read thesis.tex** - Understand overall structure
2. **Check chapters/c1/introduction.tex** - Assess problem statement clarity
3. **Review figures/** - Verify diagram quality and relevance
4. **Scan references.bib** - Check citation diversity and recency
5. **Look for:**
   - Clear research questions
   - Methodology justification
   - Results interpretation
   - Academic writing quality

**Review output:** Create structured feedback document with:
- Strengths (what works well)
- Areas for improvement
- Questions/clarifications needed
- Technical suggestions

---

## UNIQUE ASPECTS

### Phat's Thesis
- Has companion ML codebase in `src/`
- Implements graph neural network for causal discovery
- Uses `v26e_xyaug.py` as latest model version
- Outline.md contains chapter guidance notes

### Quân's Thesis
- Modular sub-file structure (c1_context, c1_problem, etc.)
- Vietnamese language with English technical citations
- Focus on industrial tool (Qodo Cover/Plus)
- Self-healing mechanism as core contribution

### Minh's Thesis
- Survey data from 48 educators
- Market positioning analysis (chart)
- "One Lesson, One Evening" design philosophy
- Agentic AI + RAG architecture

---

## NOTES

- All theses follow similar LaTeX template but have distinct content
- Minh's thesis has compiled PDF (most complete)
- Quân's thesis has most modular structure
- Phat's thesis has ML codebase and needs completion
- Bibliography files are independent per thesis
- Build requires `pdflatex`, `bibtex`, and Vietnamese fonts
