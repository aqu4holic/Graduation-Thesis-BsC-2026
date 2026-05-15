# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a graduation thesis (BSc 2026) on "Causal Discovery on Time Series Data". The project includes:
- **LaTeX thesis document** at `NGUYỄN_THÀNH_PHÁT___KLTN_T5_2026/`
- **Machine learning codebase** at `src/` implementing neural network models for causal discovery
- Participation in the **ADIA Lab Causal Discovery Challenge** via CrunchDAO platform

## Repository Structure

```
├── NGUYỄN_THÀNH_PHÁT___KLTN_T5_2026/    # LaTeX thesis document
│   ├── thesis.tex                       # Main thesis file
│   ├── chapters/                        # Chapter content (c1-c5)
│   ├── figures/                         # TikZ figures and images
│   └── references.bib                   # Bibliography
├── src/                                 # ML source code
│   ├── data/                            # Training/test data (pickle files)
│   ├── v*.py                            # Model experiment versions (v11-v30)
│   ├── *.ipynb                          # Jupyter notebooks for exploration
│   ├── evaluate.py                      # Evaluation utilities
│   └── .crunchdao/project.json          # CrunchDAO competition config
├── papers/                              # Reference papers
├── solutions/                           # External solutions for reference
└── pyproject.toml                       # Python dependencies (UV package manager)
```

## Python Environment

**Package Manager:** UV (not pip)

```bash
# Install dependencies
uv sync

# Run a Python script
uv run python src/v26e_xyaug.py

# Run Jupyter notebook server
uv run jupyter notebook

# Add a new dependency
uv add <package-name>
```

**Key Dependencies:**
- PyTorch + PyTorch Lightning for deep learning
- Crunch-cli for competition submissions
- Pandas, NumPy, scikit-learn for data processing
- NetworkX for graph operations

## Thesis Document (LaTeX)

**Location:** `NGUYỄN_THÀNH_PHÁT___KLTN_T5_2026/`

```bash
cd NGUYỄN_THÀNH_PHÁT___KLTN_T5_2026

# Build thesis (requires pdflatex/bibtex)
pdflatex thesis.tex
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex

# Or use latexmk for automatic builds
latexmk -pdf thesis.tex
```

**Chapter Files:**
- `chapters/c1/c1_introduction.tex` - Introduction
- `chapters/c2/c2_chapter.tex` - Background/Related Work
- `chapters/c3/c3_chapter.tex` - Proposed Method
- `chapters/c4/c4_chapter.tex` - Experiments
- `chapters/c5/c5_chapter.tex` - Results

## ML Code Architecture

### Core Model (v11, v26e variants)

The main model implements a **graph neural network** for causal discovery with this architecture:

1. **Input:** Time series data as edge tensors (3 channels: sorted observations, target values, multivariate kernel regression coefficients)
2. **Edge Feature Extractor:** Stem layer + 5× Conv1d blocks + pooling → 64-dim embeddings
3. **Self-Attention:** 2 layers of multi-head attention over edges
4. **Dual Output Heads:**
   - Edge head: Binary classification for adjacency (edge exists or not)
   - Node head: 8-class classification for node roles (Confounder, Collider, Mediator, etc.)

### Key Implementation Files

| File | Purpose |
|------|---------|
| `v26e_xyaug.py` | Latest working model with XY augmentation |
| `v11_structbias.py` | Core architecture with structural bias |
| `evaluate.py` | Model evaluation metrics |
| `main.ipynb` | Reference implementation from 1st place solution |

### Model Training

```python
# From src/main.ipynb or Python scripts
from v26e_xyaug import train, infer

# Training loads data from src/data/
# Uses cached dataset at /mnt/thesis/dataset_cache/
```

**Training Parameters (see main.ipynb):**
- `N_OBS = 1000` - Number of observations per sample
- `N_KERNEL = 1000` - Kernel samples for multivariate regression
- `D_MODEL = 64` - Embedding dimension
- `N_CLASSES = 8` - Node classification classes
- `MAX_EPOCHS = 20` - Training epochs
- Batch size: 64 (training), 32 (inference)

## CrunchDAO Competition Integration

**Configuration:** `src/.crunchdao/project.json`
- Competition: `causality-discovery`
- Project name: `graduation-thesis`

```bash
cd src

# Setup CrunchDAO environment (downloads competition data)
crunch setup-notebook causality-discovery <token> --no-data

# Test locally
crunch.test(no_determinism_check=True, force_first_train=False)

# Submit to competition via notebook interface
```

**Data Files (downloaded by crunch-cli):**
- `data/X_train.pickle` - Training features (1.5GB)
- `data/y_train.pickle` - Training labels
- `data/X_test_reduced.pickle` - Test features (reduced set)
- `data/y_test_reduced.pickle` - Test labels (reduced set)

## Cloud Infrastructure

**Modal Integration:** `src/modal_volume.py`

Uses Modal cloud storage for dataset caching at `/mnt/thesis/dataset_cache/`

## Common Commands

```bash
# Thesis build
cd NGUYỄN_THÀNH_PHÁT___KLTN_T5_2026 && latexmk -pdf thesis.tex

# Run ML experiment
uv run python src/v26e_xyaug.py

# Jupyter for exploration
uv run jupyter notebook src/<notebook>.ipynb

# Update dependencies after pyproject.toml changes
uv sync
```

## Data Flow

1. Raw time series → `build_edge_tensor()` → Edge tensors (3 channels)
2. Multivariate kernel regression computes coefficients for channel 3
3. Edge tensors fed to neural network for edge/node classification
4. Post-processing: `transform_proba_to_DAG()` ensures acyclic output
5. Final adjacency matrix → CrunchDAO submission format
