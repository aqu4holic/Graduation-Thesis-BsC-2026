# Master Review: Nguyễn Hoàng Quân's Thesis

**Thesis:** Research and Improvement of Qodo Cover Automated Test Generation  
**Reviewer:** Nguyễn Thanh Phát  
**Date:** 2026  
**Total Review Time:** ~4 hours

---

## EXECUTIVE SUMMARY

This is a **strong software engineering thesis** with excellent experimental validation. The core contribution - a self-healing mechanism for LLM-generated tests - is well-motivated, technically sound, and convincingly demonstrated across 10 open-source projects.

**Overall Rating:** ⭐⭐⭐⭐☆ (4/5) - Can reach ⭐⭐⭐⭐⭐ after fixing C2-C3 duplication

---

## CHAPTER-BY-CHAPTER SCORES

| Chapter | Rating | Key Strength | Key Issue |
|---------|--------|--------------|-----------|
| **C1: Introduction** | ⭐⭐⭐⭐☆ | Clear problem statement | Content duplication |
| **C2: Background** | ⭐⭐⭐⭐⭐ | Comprehensive theory | Duplicates with C3 |
| **C3: Solution** | ⭐⭐⭐☆☆ | Excellent technical detail | **CRITICAL: Duplicates C2** |
| **C4: Experiments** | ⭐⭐⭐⭐⭐ | Outstanding validation | Minor cleanup needed |
| **Conclusion** | ⭐⭐⭐⭐☆ | Honest, forward-looking | Could add more specifics |

---

## MAJOR ISSUES (Must Fix)

### 🔴 CRITICAL: C2-C3 Content Duplication

**Problem:** c3_analysis_and_evaluation.tex contains ~88 lines nearly identical to c2_analysis_and_evaluation.tex

**Impact:** 
- Reader encounters same content twice
- Suggests poor editing
- Wastes thesis space

**Fix:**
1. DELETE duplicate content from c3_analysis_and_evaluation.tex
2. REPLACE with 5-line summary referencing C2:
   ```latex
   \subsection{Tóm tắt hạn chế Qodo Cover}
   Như đã phân tích chi tiết ở Mục \ref{sec:problems_of_qodo}, 
   Qodo Cover gặp ba vấn đề chính: (1) cơ chế xử lý theo lô, 
   (2) lãng phí tài nguyên, và (3) dừng phát triển CLI.
   ```

**Estimated fix time:** 2 hours

---

### 🟡 MODERATE: Miscellaneous Content Duplication

**Issues:**
1. Qodo Cover discontinuation date (15/06/2025) appears 3+ times
2. CLI limitation explanation repeated
3. Batch-processing critique appears twice

**Fix:** Consolidate to C1 or C2, reference elsewhere

---

### 🟡 MODERATE: Commented Code Cleanup

**Files with extensive commented content:**
- c3_chapter.tex (30 lines of template comments)
- c2_ai_agent.tex (15 lines of LLM descriptions)
- c3_solution.tex (commented figure code)
- c4_chapter.tex (30 lines of template comments)
- c4_analysis_and_evaluation.tex (commented analysis)

**Fix:** Remove before final submission

---

## MINOR ISSUES (Should Fix)

### Figure References
- \ref{fig:batch_processing} referenced in C1, defined in C2 - check if exists
- \ref{fig:self_healing_pipeline} referenced but figure code commented out

### Citation Verification
- Duplicate keys in references.bib (lines 69, 182)
- Verify \cite{qodo} exists
- Verify Lewis 2021 and Tran 2025 in conclusion exist

### Writing Style
- Some very long sentences (especially in c1_problem.tex)
- Consider breaking MAPE-K paragraph into bullet points

---

## STRENGTHS (Maintain & Highlight)

### 1. Experimental Design (C4) ⭐⭐⭐⭐⭐
- **10 diverse projects:** Flask, Django REST, HanLP, Gymnasium, OpenAI-Python, LocalStack, Locust, Pipenv, Scrapy, tqdm
- **3 LLMs tested:** DeepSeek-V3.2, Qwen3 Coder, GPT-OSS-120B
- **Clear metrics:** Line coverage + branch coverage
- **Quantified results:** 5-36% line, 10-47% branch improvements

### 2. Technical Detail (C3) ⭐⭐⭐⭐⭐
- **Dual-trigger self-healing:** Execution failure OR coverage plateau
- **3-layer prompt architecture:** System + Dynamic context + Execution anchor
- **6 concrete code examples:** Before/after comparisons
- **Edge case handling:** Flaky test detection, useless test filtering

### 3. Background Foundation (C2) ⭐⭐⭐⭐⭐
- **pytest mechanisms:** Discovery, Fixture, Assertion
- **AI Agent components:** Agent, Planning, Memory, Tools
- **Self-healing evolution:** Infrastructure → UI → Code level
- **MAPE-K loop:** Detailed explanation

### 4. Honest Assessment (C4/Conclusion) ⭐⭐⭐⭐⭐
- Acknowledges regression cases
- Discusses non-determinism issues
- Lists 4 genuine limitations
- Concrete future directions

---

## COMPARISON WITH PEER THESES

| Aspect | Quân | Minh | Phat (me) |
|--------|------|------|-----------|
| **Domain** | Software testing | EdTech platform | Causal discovery |
| **Language** | Vietnamese | English | English |
| **Structure** | Modular (34 .tex) | Modular (22 .tex) | Medium (25 files) |
| **Figures** | 17 workflow diagrams | Survey charts + tech | TikZ diagrams |
| **Citations** | 70+ (LLM/testing) | 25 (education) | 32 (causal ML) |
| **Data** | 10 open-source projects | 48-educator survey | 47K ML datasets |
| **Models** | 3 LLMs | 1 AI system | 1 neural architecture |
| **Results** | Coverage % | Survey analysis | Accuracy % |
| **Code examples** | 6 Python listings | Minimal | Minimal |

**Unique strengths of Quân's thesis:**
- Most modular structure
- Strongest experimental validation
- Best code-level technical detail
- Honest about limitations

---

## RECOMMENDATIONS FOR AUTHOR

### Immediate Actions (Before Defense)

**Week 1:**
1. [ ] **CRITICAL:** Fix C2-C3 duplication (2 hours)
2. [ ] Remove all commented template text (1 hour)
3. [ ] Verify/fix all figure references (30 min)
4. [ ] Fix duplicate .bib entries (30 min)

**Week 2:**
5. [ ] Add algorithm pseudocode for self-healing loop (1 hour)
6. [ ] Break up long sentences in C1 (1 hour)
7. [ ] Consolidate regression explanations (30 min)
8. [ ] Final proofreading (2 hours)

### Optional Improvements
- Add bar chart comparing 3 LLMs
- Add Extension screenshot
- Add cost analysis (API tokens)
- Add threats to validity section

---

## WHAT I LEARNED (FOR MY THESIS)

### Technical:
1. **Before/after code comparisons** - very effective for showing improvement
2. **Multi-model validation** - strengthens claims significantly
3. **Dual-trigger design** - sophisticated approach to edge cases
4. **Prompt engineering architecture** - systematic breakdown of layers

### Writing:
1. **Modular structure** - easier to maintain than single file
2. **Honest limitations** - strengthens rather than weakens thesis
3. **Concrete code examples** - makes abstract concepts tangible
4. **Real-world dataset** - 10 production projects very convincing

### Organization:
1. **Detailed background** - C2 is comprehensive reference
2. **Separate experiment chapter** - C4 stands alone as validation
3. **Clear chapter roadmap** - helps reader navigate

---

## FINAL VERDICT

**Ready for defense?** Almost - fix C2-C3 duplication first

**Estimated time to ready:** 5-6 hours

**Strongest aspect:** Experimental validation (C4)

**Weakest aspect:** Content organization (C2-C3 duplication)

**Recommendation:** Fix duplication issues, then this is a ⭐⭐⭐⭐⭐ thesis

---

## REVIEW FILES CREATED

| File | Location | Content |
|------|----------|---------|
| Chapter 1 Review | c1/REVIEW.md | Context, problem, purpose analysis |
| Chapter 2 Review | c2/REVIEW.md | Background, pytest, AI Agent, self-healing |
| Chapter 3 Review | c3/REVIEW.md | **C2-C3 duplication highlighted** |
| Chapter 4 Review | c4/REVIEW.md | Experimental results analysis |
| Conclusion Review | chapters/REVIEW-CONCLUSION.md | Summary and future work |
| Master Review | MASTER-REVIEW.md | This file |

---

**Reviewer Signature:** Nguyễn Thanh Phát  
**Date:** 2026  
**Contact:** Available for follow-up questions
