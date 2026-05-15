# Thesis Knowledge Base - NGUYỄN HOÀNG QUÂN

**Thesis:** Research and Improvement of Qodo Cover Automated Test Generation  
**Topic:** Qodo Plus - Self-healing AI test generation with LLM feedback  
**Language:** Vietnamese (with English technical citations)  
**Status:** Complete structure, modular chapter organization  

---

## OVERVIEW

This thesis documents **Qodo Plus** - an enhanced version of the Qodo Cover tool integrating a **self-healing mechanism** for automated test generation. The research addresses the key limitation of existing LLM-based test generation: rigid linear workflows that discard failing tests.

**Core Innovation:** Feedback loop that extracts error messages, prompts LLM for local fixes, archives error history, and generates new tests inheriting past context.

**Results:** Experiments on 10 top-tier Python GitHub projects show complete resolution of instability, increased line/branch coverage, and optimized API costs.

---

## THESIS STRUCTURE (34 .tex files)

```
Nguyễn_Hoàng_Quân___KLTN_T5_2026/
├── thesis.tex                          # Main document (report, 13pt)
├── cover.tex                          # Double-border TikZ cover (VI/EN)
├── references.bib                     # 673 lines, 70+ citations
├── empty.tex                          # Placeholder
└── chapters/
    ├── acknowledgement.tex            # Lời cảm ơn
    ├── assurance.tex                  # Lời cam đoan
    ├── abtract_vi.tex                 # Tóm tắt tiếng Việt
    ├── abtract_en.tex                 # Abstract English
    ├── glossary.tex                   # Thuật ngữ
    ├── conclusion.tex                 # Kết luận
    ├── introduction.tex               # Legacy intro (unused)
    ├── method.tex                     # Legacy method (unused)
    ├── evaluation.tex                 # Legacy evaluation (unused)
    │
    ├── c1/                            # Chapter 1: Đặt vấn đề
    │   ├── c1_introduction.tex        # Main wrapper
    │   ├── c1_context.tex             # Bối cảnh
    │   ├── c1_problem.tex             # Phát biểu bài toán
    │   └── c1_purpose.tex             # Mục tiêu
    │
    ├── c2/                            # Chapter 2: Cơ sở lý thuyết
    │   ├── c2_chapter.tex             # Main wrapper
    │   ├── c2_automation_testing.tex  # Kiểm thử tự động
    │   ├── c2_ai_agent.tex            # AI Agent
    │   ├── c2_self_healing.tex        # Self-healing
    │   └── c2_analysis_and_evaluation.tex  # Đánh giá
    │
    ├── c3/                            # Chapter 3: Giải pháp
    │   ├── c3_chapter.tex             # Main wrapper
    │   ├── c3_approach.tex              # Hướng tiếp cận
    │   ├── c3_solution.tex              # Giải pháp chi tiết
    │   └── c3_analysis_and_evaluation.tex  # Phân tích
    │
    ├── c4/                            # Chapter 4: Thực nghiệm
    │   ├── c4_chapter.tex             # Main wrapper
    │   ├── c4_dataset.tex             # Dữ liệu thực nghiệm
    │   ├── c4_test_script.tex         # Kịch bản kiểm thử
    │   ├── c4_test_result.tex         # Kết quả thử nghiệm
    │   └── c4_analysis_and_discussion.tex  # Phân tích thảo luận
    │
    └── c5/                            # Chapter 5 (unused)
        ├── c5_chapter.tex             # Classification wrapper
        ├── c5_classification.tex      # (not in main thesis.tex)
        └── c5_detection.tex           # (not in main thesis.tex)
```

---

## FIGURES (17 images)

```
figures/
├── uet.jpg                            # UET logo
├── qodo.png, qodo_plus.png            # Tool logos
├── Workflow.png                       # General workflow
├── Workflow_self_healing.png          # Self-healing loop
├── simple_workflow.png                # Simplified workflow
├── simple_workflow_qodoplus.png       # Qodo Plus workflow
├── MAPE-K.png                         # MAPE-K control loop
├── LLM_Agent.png                      # Agent architecture
├── transformer.png                    # Transformer diagram
├── prompt.png                         # Prompt engineering
├── flow.png                           # Process flow
├── new_flows.png                      # Updated flows
├── main_change.png                    # Key modifications
├── input_output.png                   # I/O specification
├── intro.png                          # Introduction figure
└── auto_test_coverage.png             # Coverage metrics
```

---

## LATEX CONVENTIONS

### Document Class
```latex
\documentclass[a4paper,13pt]{report}
\usepackage[utf8]{vietnam}
\changefontsizes{13pt}
```

### Vietnamese Localization
```latex
\renewcommand*{\ALG@name}{Thuật toán}  % Algorithm → Thuật toán
\renewcommand{\cftchappresnum}{Chương }  % Chapter → Chương
```

### Custom Code Styling
```latex
\definecolor{bg_gray}{RGB}{242, 242, 235}
\lstset{
  backgroundcolor=\color{bg_gray},
  numbers=left,
  numberstyle=\small\color{gray},
}
```

### TikZ Cover Design
```latex
\usetikzlibrary{calc}
% Double border frame with rounded corners
\draw[line width=3pt] ...
\draw[line width=1pt] ...
```

---

## BIBLIOGRAPHY PATTERNS

**Style:** `unsrt` (numbered by appearance)  
**Entries:** 70+ references

### Key Citation Categories

| Category | Examples |
|----------|----------|
| **LLM Papers** | GPT-4, ChatGPT, Claude, Gemini, DeepSeek, Qwen |
| **Testing Research** | ICSE, IEEE TSE, FSE papers |
| **AI Agents** | ReAct, self-healing systems |
| **Coverage** | Line coverage, branch coverage, mutation testing |
| **Open Source** | Flask, Django REST, Scrapy references |

**Citation Keys:** Semantic format
- `wang2024hitshighcoveragellmbasedunit`
- `fan2023automatedrepairprogramslarge`
- `vaswani2023attentionneed`

---

## CHAPTER ORGANIZATION

| Chapter | Title | Content |
|---------|-------|---------|
| **C1** | Đặt vấn đề | Context, problem, objectives |
| **C2** | Cơ sở lý thuyết | Automated testing, AI Agents, Self-healing, MAPE-K |
| **C3** | Giải pháp Qodo Plus | Approach, architecture, implementation |
| **C4** | Thực nghiệm | Dataset, test scripts, results, analysis |
| **Conclusion** | Kết luận | Summary, limitations, future work |

**Note:** Chapter 5 files exist but are not included in `thesis.tex`. The thesis uses a 4-chapter structure.

---

## BUILD COMMANDS

```bash
cd /home/phatnguyen11/code/Graduation-Thesis-BsC-2026/Nguyễn_Hoàng_Quân___KLTN_T5_2026/

# Build
latexmk -pdf thesis.tex

# Or manual
pdflatex thesis.tex
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex
```

---

## PEER REVIEW CONTEXT

When this thesis is reviewed by peers:

**Strengths to note:**
- Highly modular structure (sub-files per chapter section)
- Rich figure set (17 technical diagrams)
- Comprehensive bibliography (70+ references)
- Clear problem-solution-evaluation flow

**Areas for feedback:**
- Vietnamese technical writing quality
- Clarity of self-healing mechanism explanation
- Experimental methodology rigor
- Figure caption completeness

---

## COMPARISON WITH PEER THESES

| Aspect | Quân (This) | Phat | Minh |
|--------|-------------|------|------|
| **Language** | Vietnamese | English | English |
| **Structure** | Most modular | Medium | Clean 4-chapter |
| **Figures** | 17 workflow/technical | TikZ diagrams | Survey charts |
| **Citations** | 70+ (LLM/testing) | 32 (causal ML) | 25 (education/AI) |
| **Domain** | Software testing | Causal discovery | EdTech platform |

---

## SUPERVISORS

- **TS. Nguyễn Văn Sơn** (Dr. Nguyen Van Son) - Primary
- **PSG.TS. Võ Đình Hiếu** (Assoc. Prof. Vo Dinh Hieu) - Co-supervisor
- Institution: Vietnam National University, Hanoi - UET
