# Thesis Knowledge Base - LÊ HOÀNG MINH

**Thesis:** Agentic AI Teaching Slides Development Platform (Edlora)  
**Topic:** AI-powered lesson development combining storytelling, gamification, and RAG  
**Language:** English  
**Status:** Complete with compiled PDF and survey data  

---

## OVERVIEW

This thesis presents **Edlora** - an Agentic AI Lesson Development Environment (LDE) that addresses the teacher workload crisis by enabling educators to generate curriculum-aligned, narrative-driven teaching presentations through natural language interaction.

**Core Vision:** "One Lesson, One Evening" - teachers create complete lessons in under 1 hour

**Key Innovation:** Multi-Agent AI workflow using LangGraph (12 nodes: Lesson Director + Lesson Developer agents) with RAG pipeline for curriculum grounding.

---

## THESIS STATUS: COMPLETE

✅ **Compiled PDF exists:** `thesis.pdf`  
✅ **Full LaTeX auxiliary files:** .aux, .toc, .lof, .lot, .bbl, .blg  
✅ **Survey data:** 48 educators  
✅ **Testing metrics:** 86.56% backend coverage, RAG benchmarks  

---

## THESIS STRUCTURE

```
LÊ_HOÀNG_MINH___KLTN_T5_2026/
├── thesis.tex                         # Main document (177 lines)
├── thesis.pdf                         # ✅ COMPILED OUTPUT
├── thesis.aux, .toc, .lof, .lot       # Table of contents/lists
├── thesis.bbl, .blg                   # BibTeX output
├── thesis.synctex.gz                  # SyncTeX for PDF navigation
├── thesis.out                         # hyperref bookmarks
├── cover.tex                          # English cover (TikZ double-frame)
├── references.bib                     # 183 lines, 25 citations
├── empty.tex                          # Placeholder
└── chapters/
    ├── acknowledgement.tex            # Acknowledgements (minimal)
    ├── assurance.tex                  # Academic integrity (roman nums)
    ├── abtract_en.tex                 # English abstract
    ├── abtract_vi.tex                 # Vietnamese abstract
    ├── glossary.tex                   # Abbreviations (BiLSTM, BO)
    ├── conclusion.tex                 # Conclusion & future work
    ├── introduction.tex               # Legacy (unused)
    │
    ├── c1/c1_introduction.tex         # Chapter 1: Introduction
    │                                    # Survey of 48 teachers
    │                                    # Market gap analysis
    │                                    # "3 Rules" of Edlora
    │
    ├── c2/c2_chapter.tex              # Chapter 2: Background
    │                                    # Storytelling + Gamification
    │                                    # RAG equation
    │                                    # WSGI vs ASGI architecture
    │
    ├── c3/c3_chapter.tex              # Chapter 3: Edlora Platform
    │                                    # 5 Use Cases (UC-01 to UC-05)
    │                                    # 7 AI behavioral rules
    │                                    # 12-node multi-agent graph
    │                                    # ER diagram
    │
    └── c4/c4_chapter.tex              # Chapter 4: Results
                                         # Testing pyramid metrics
                                         # LLM-as-Judge evaluation
                                         # RAG benchmark (233 lines)
```

---

## FIGURES (18 images)

```
figures/
├── uet.jpg                                    # UET logo
├── transformer_architecture.png               # Transformer diagram
├── rag_architecture.png                      # RAG pipeline
├── use_case_diagram.png                      # 5 UCs diagram
├── er_diagram.png                            # Database schema
├── agentic_graph.png                         # 12-node LangGraph
├── edlora_product_positioning.png           # Market positioning
│
# Survey Result Charts (48 educators):
├── how_many_slides_are_created_manually.png
├── why_dont_teachers_make_their_own_slides.png
├── where_do_slides_come_from.png
├── issue_with_premade_slides.png
├── issue_with_selfmade_slides.png
├── how_often_do_teachers_put_storytelling_elements_in.png
│
└── [additional technical diagrams]
```

---

## SURVEY DATA SUMMARY (n=48)

### Slide Creation Frequency
- 29.8% make 1-2 new presentations/week
- 21.3% make zero new slides/week

### Barriers to Creation
- 39.5% lack of time
- 27.9% lack technical skills

### Slide Sources
- 46.5% minor edits of old files
- 27.9% bought/shared files
- 11.6% built from scratch

### Self-Made Slides - Time Allocation (n=25)
- 72% story and flow planning
- 48% interactive academic games
- 32% finding media
- 32% layout and colors

### Storytelling Usage (n=43)
- 53.5% use when have time
- 27.9% use frequently
- Only 1 teacher felt unnecessary

---

## EDLORA'S "3 RULES"

1. **"One Lesson, One Evening"** - Complete lesson in <1 hour
2. **"One Lesson, One Knowledge Journey"** - Story elements + teaching frameworks
3. **"One Creation Method"** - Natural language interface (no complex technical skills)

---

## TECHNICAL STACK

### AI/ML Layer
- GPT-4o (primary model)
- LangGraph + LangChain (multi-agent orchestration)
- OpenAI Embeddings (text vectorization)
- pgvector (PostgreSQL vector extension)
- MinerU (PDF parsing for curriculum ingestion)

### Frontend
- Next.js 14 + React + TypeScript
- Tailwind CSS + MUI (Material-UI)
- TipTap (block-based editor)
- Zustand (state management)
- TanStack Query (data fetching)

### Backend
- FastAPI + Uvicorn
- Pydantic + SQLModel
- PostgreSQL + Alembic (migrations)

### Infrastructure
- Clerk (authentication)
- Cloudflare R2 (object storage)
- Supabase (database hosting)

### Testing & Evaluation
- pytest (backend unit tests)
- vitest (frontend unit tests)
- Playwright (E2E testing)
- Storybook + Chromatic (component testing)
- LangSmith (LLM observability)
- OpenEvals (RAG evaluation)

### DevOps
- Docker (containerization)
- GitHub Actions (CI/CD)
- Render (backend deployment)
- Vercel (frontend deployment)

---

## LATEX CONVENTIONS

### Document Class
```latex
\documentclass[a4paper,13pt]{report}
\usepackage[T5]{fontenc}           % Vietnamese font encoding
\usepackage[utf8]{inputenc}
\usepackage[unicode]{hyperref}
```

### Chapter Formatting
```latex
\renewcommand{\cftchappresnum}{Chapter }
\AtBeginDocument{\addtolength\cftchapnumwidth{\widthof{\bfseries Chapter }}}

\titleformat{\chapter}[display]   
{\normalfont\huge\bfseries}{\chaptertitlename\ \thechapter}{0pt}{\LARGE}
```

### Custom Tables
```latex
\usepackage{longtable}             % Multi-page tables
\usepackage{threeparttable}        % Notes under tables
% Use case specs: p{3.5cm}|p{11.5cm} column widths
```

### TikZ Cover Design
```latex
\usetikzlibrary{calc}
% Double-frame border with positioning calculations
\draw[line width=3pt] 
  ($ (current page.north west) + (25mm,-25mm) $) ...
```

---

## BIBLIOGRAPHY (25 references)

### Citation Categories

| Category | Key Works |
|----------|-----------|
| **Cognitive Science** | Schank (1995), Graeber (2024), Bower (1969) - storytelling retention |
| **Gamification** | Hamari (2014), Jihadillah (2025) - 12.9% failure rate reduction |
| **AI/ML** | Lewis et al. (2020) RAG, Yao (2022) ReAct |
| **Education** | Kyriacou (2001) teacher stress, Mayer (2014) multimedia |
| **Labor Market** | WEF (2023) Future of Jobs Report |

---

## CHAPTER ORGANIZATION

| Chapter | Title | Key Content |
|---------|-------|-------------|
| Front Matter | Cover, Abstracts, Glossary | Roman pagination (i, ii, iii...) |
| **Chapter 1** | Introduction | Survey (n=48), problem analysis, market gap, Edlora vision |
| **Chapter 2** | Background | Storytelling + gamification pedagogy, RAG, architecture |
| **Chapter 3** | Edlora Platform | 5 UCs, 7 AI rules, 12-node agent graph, ER diagram |
| **Chapter 4** | Results | 86.56% coverage, LLM-as-Judge, RAG benchmarks |
| **Conclusion** | Future Work | Short-term (latency, lesson plans), Long-term (multimodal, personalization) |

---

## TESTING METRICS

### Backend Coverage (Testing Pyramid)
- **Unit tests:** 86.56% line coverage
- **Integration tests:** API endpoint validation
- **E2E tests:** Playwright user flows

### LLM-as-Judge Evaluation
- Correctness scoring by GPT-4
- RAG benchmark metrics:
  - Context Precision
  - Context Recall
  - Answer Relevance

---

## BUILD COMMANDS

```bash
cd /home/phatnguyen11/code/Graduation-Thesis-BsC-2026/LÊ_HOÀNG_MINH___KLTN_T5_2026/

# Build (PDF already exists)
latexmk -pdf thesis.tex

# Or manual 4-pass
pdflatex thesis.tex
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex

# Clean
latexmk -c
```

---

## PEER REVIEW CONTEXT

When this thesis is reviewed by peers:

**Strengths to note:**
- Complete compiled PDF available
- Survey data from 48 real educators
- Market positioning analysis
- Clear "3 Rules" product vision
- Comprehensive testing metrics
- Professional use case documentation

**Areas for feedback:**
- Clarity of AI agent workflow explanation
- RAG implementation technical depth
- Survey methodology documentation
- Comparison with existing EdTech tools

---

## COMPARISON WITH PEER THESES

| Aspect | Minh (This) | Phat | Quân |
|--------|-------------|------|------|
| **Language** | English | English | Vietnamese |
| **Status** | ✅ Complete PDF | Partial | Complete structure |
| **Data** | 48-educator survey | ML competition | Open source projects |
| **Figures** | Survey charts + tech | TikZ diagrams | Workflow diagrams |
| **Citations** | 25 (education/AI) | 32 (causal ML) | 70+ (LLM/testing) |
| **Domain** | EdTech platform | Causal discovery | Software testing |

---

## SUPERVISORS

- **TS. Nguyễn Văn Sơn** (Dr. Nguyen Van Son)
- **PSG.TS. Võ Đình Hiếu** (Assoc. Prof. Vo Dinh Hieu)
- Institution: Vietnam National University, Hanoi - UET

---

## NOTES FOR PEER REVIEWERS

This thesis serves as a **high-quality template** for:
- Software engineering thesis documentation
- Use case specification (5 UCs with detailed flows)
- Survey-based research methodology
- System architecture diagrams
- Multi-agent AI workflow documentation

The compiled PDF can be used as a reference for formatting, figure placement, and academic writing style.
