# Chapter 3 Review: Giải pháp Qodo Plus

**Reviewer:** Nguyễn Thanh Phát  
**Date:** 2026  
**Files Reviewed:** c3_chapter.tex, c3_approach.tex, c3_solution.tex, c3_analysis_and_evaluation.tex

---

## 1. OVERALL ASSESSMENT

**Strength:** ⭐⭐⭐☆☆ (3/5)

Chapter 3 has significant content duplication with Chapter 2 that needs immediate attention. However, the solution description itself is technically sound and well-structured once unique content is isolated.

**CRITICAL ISSUE:** ~88 lines in c3_analysis_and_evaluation.tex are nearly identical to c2_analysis_and_evaluation.tex

---

## 2. DETAILED SECTION REVIEWS

### 2.1 c3_chapter.tex - Main wrapper

**Assessment:** Standard chapter wrapper file.

**Content:**
- ✅ Appropriate comment template from thesis guidelines
- ✅ Clear chapter objective statement
- ✅ Correct `\input{}` sequence for sub-files

**Note:** Lines 3-33 contain commented template text. Consider removing before final submission.

---

### 2.2 c3_approach.tex - Hướng tiếp cận

**Strengths:**
- ✅ Clear 4 principles of Qodo Plus design:
  1. Local feedback loop for error fixing
  2. Global fallback with context synthesis
  3. Edge case handling and resource control
  4. Core architecture extension
- ✅ Logical progression from problem to solution principles
- ✅ Good connection to C1 problem statement

**Minor Issues:**
1. **Reference to C1:** Line 2 says "From issues mentioned in Section \ref{sec:problems_of_qodo}" - verify this label exists (it may be in C2)
2. **Principle 3 truncation:** Line 7 seems to end abruptly: "Hệ thống cần được trang bị các cơ chế phòng vệ" - check if complete

**Suggestions:**
1. **Quantify principle 2:** "max-fix-attempts" mentioned - give the actual number (3? 5?)
2. **Add visual:** A simple diagram showing 4 principles as a foundation could help
3. **Connect to MAPE-K:** Briefly mention how these 4 principles map to Monitor/Analyze/Plan/Execute

---

### 2.3 c3_solution.tex - Thiết kế giải pháp

**Strengths:**
- ✅ Excellent 4-component structure:
  1. Self-healing workflow (local + global)
  2. Branch coverage improvement via prompt engineering
  3. Operational optimization and edge case handling
  4. Extension development
- ✅ Specific technical details throughout
- ✅ Good use of code listings (Python syntax highlighted)
- ✅ Multiple figures referenced appropriately

**Technical Highlights:**
- Local self-healing with two triggers: execution failure AND coverage plateau
- Detailed prompt engineering architecture (3 layers)
- Flaky test detection mechanism
- Extension/IDE integration approach

**Issues:**

1. **Figure placement:** Lines 14-19 have commented-out figure:
   ```latex
   % \begin{figure}[htpb]
   %     \centering
   %     \includegraphics[width=0.9\textwidth]{figures/new_flows.png}
   % ...
   ```
   - Either uncomment if figure exists, or remove entirely

2. **Missing figure:** Line references `\ref{fig:self_healing_pipeline}` but it's commented out

3. **Code style definition:** Lines 20-45 define Python style in the middle of content. Consider:
   - Moving to thesis.tex preamble
   - Or using consistent style with other code listings

4. **Long paragraph:** Lines 86-94 (Self-healing prompt section) is very long

**Specific Technical Feedback:**

**Self-healing triggers (excellent design):**
- Execution trigger: exit code ≠ 0
- Coverage trigger: exit code = 0 but no coverage increase
- This dual-trigger approach is sophisticated - highlight it more prominently

**Prompt engineering 3-layer architecture:**
1. System prompt (role + constraints + heuristic logic)
2. Dynamic context injection (Jinja2 templates)
3. Task execution anchor

This is a strong contribution - consider making this a subsection or even a figure

**Suggestions for improvement:**

1. **Add algorithm pseudocode:** The self-healing workflow could benefit from Algorithm environment:
   ```latex
   \begin{algorithm}
   \caption{Local Self-Healing Loop}
   \begin{algorithmic}
   \FOR{each test case}
       \STATE Run pytest
       \IF{exit $\neq$ 0 OR coverage unchanged}
           \STATE Trigger self-healing
           \FOR{i = 1 to max\_attempts}
               \STATE Call LLM with error context
               \STATE Update test file
               \STATE Re-run pytest
               \IF{pass AND coverage increased}
                   \STATE Save test; \textbf{break}
               \ENDIF
           \ENDFOR
       \ENDIF
   \ENDFOR
   \end{algorithmic}
   \end{algorithm}
   ```

2. **Prompt template example:** Show actual prompt text (anonymized) to illustrate the approach

3. **Extension screenshot:** If available, a screenshot of the IDE extension would strengthen this section significantly

---

### 2.4 c3_analysis_and_evaluation.tex - Các vấn đề hiện tại của Qodo Cover

**CRITICAL ISSUE:** This file contains ~88 lines that are nearly IDENTICAL to c2_analysis_and_evaluation.tex

**Duplicated content includes:**
- Qodo Cover description (lines 1-18)
- 3-phase workflow description (lines 21-61)
- Limitations section with batch-processing critique (lines 59-88)

**Impact:**
- Reader will encounter the same content twice
- Wastes space in thesis
- Suggests poor editing/planning

**Recommendation:** 
1. **DELETE** the duplicated content from c3_analysis_and_evaluation.tex
2. **REPLACE** with brief summary paragraph:
   ```latex
   \subsection{Tóm tắt hạn chế Qodo Cover}
   Như đã phân tích chi tiết ở Mục \ref{sec:problems_of_qodo}, 
   Qodo Cover gặp ba nhóm vấn đề chính: (1) cơ chế xử lý 
   theo lô gây mất ổn định, (2) lãng phí tài nguyên do 
   rollback tức thời, và (3) dừng phát triển kèm hạn chế CLI.
   Các vấn đề này là cơ sở để thiết kế giải pháp Qodo Plus.
   ```
3. **Keep only unique content** in c3_analysis_and_evaluation.tex

**Unique content in c3_analysis_and_evaluation.tex:**
- Check if there are any sections NOT in c2 version
- If entire file is duplicate, consider deleting file entirely and removing `\input{}`

---

## 3. CROSS-CUTTING ISSUES

### 3.1 C2-C3 Duplication Map

| Content | C2 File | C3 File | Action |
|---------|---------|---------|--------|
| Qodo Cover intro | c2_analysis (lines 1-6) | c3_analysis (lines 1-6) | Keep in C2, delete C3 |
| 3-phase workflow | c2_analysis (lines 27-61) | c3_analysis (lines 21-56) | Keep in C2, delete C3 |
| Limitations | c2_analysis (lines 59-88) | c3_analysis (lines 59-88) | Keep in C2, delete C3 |
| Solution approach | - | c3_approach | Keep |
| Technical design | - | c3_solution | Keep |

### 3.2 Figure Reference Check

| Figure | Referenced In | File Exists? | Status |
|--------|---------------|--------------|--------|
| \ref{fig:self_healing_pipeline} | c3_solution | new_flows.png? | ⚠️ Check |
| \ref{fig:Workflow_self_healing} | c3_solution | ✅ | Good |
| \ref{fig:simple_workflow_qodoplus} | c3_solution | ✅ | Good |
| \ref{fig:prompt} | c3_solution | ✅ | Good |
| \ref{lst:PRNG} | c3_solution | Code in file | Good |
| \ref{lst:PRNG_plus} | c3_solution | Code in file | Good |
| \ref{lst:sync_error} | c3_solution | Code in file | Good |
| \ref{lst:sync_plus} | c3_solution | Code in file | Good |
| \ref{lst:dictionary_error} | c3_solution | Code in file | Good |
| \ref{lst:dictionary_plus} | c3_solution | Code in file | Good |

### 3.3 Code Listing Quality

**Good aspects:**
- Python syntax highlighting defined
- Clear before/after comparisons (Cover vs Plus)
- Line numbers for easy reference
- Captions explain the key difference

**Suggestions:**
1. Consider highlighting key changed lines with comments or color
2. Add short annotation explaining why the change fixes the issue

---

## 4. RECOMMENDATIONS

### Critical fixes (before defense):
1. [ ] **CRITICAL:** Remove/condense duplicate content in c3_analysis_and_evaluation.tex
2. [ ] Uncomment or remove commented figure code (c3_solution.tex lines 14-19)
3. [ ] Fix or remove \ref{fig:self_healing_pipeline} reference
4. [ ] Verify Python style definition location

### Improvements:
1. [ ] Add algorithm pseudocode for self-healing loop
2. [ ] Show actual (anonymized) prompt example
3. [ ] Add Extension screenshot if available
4. [ ] Break up long paragraph at c3_solution.tex lines 86-94
5. [ ] Add MAPE-K mapping for the 4 principles

### Questions for author:
1. Is figures/new_flows.png available or should reference be removed?
2. What is the actual max_fix_attempts value used?
3. Are the code listings in c3_solution.tex actual generated examples or simplified?

---

## 5. COMPARISON WITH MY THESIS (PHAT)

| Aspect | Quân's C3 | My C3 | Notes |
|--------|-----------|-------|-------|
| **Structure** | 4 sub-files | Single file | Quân more modular |
| **Length** | ~250 lines (with dupes) | 307 lines | Comparable |
| **Duplication** | ~88 lines duplicate | None | Major issue for Quân |
| **Figures** | 4 figures + 6 code listings | Multiple TikZ | Rich technical detail |
| **Algorithms** | Prose description | Pseudocode | Both could use more |
| **Code examples** | 6 Python listings | Minimal | Quân stronger here |

**What I can learn:**
- Excellent before/after code comparisons
- Clear 3-layer prompt engineering architecture
- Dual-trigger design (execution + coverage)
- Strong practical focus on IDE extension

---

## 6. FINAL VERDICT

Chapter 3 has excellent technical content but is severely undermined by the C2-C3 duplication. Once resolved, this will be a strong chapter.

**Technical quality:** ⭐⭐⭐⭐⭐ (5/5)
**Organization:** ⭐⭐⭐☆☆ (3/5)
**Uniqueness:** ⭐⭐☆☆☆ (2/5) - due to duplication

**Overall:** ⭐⭐⭐☆☆ (3/5) - Can become ⭐⭐⭐⭐⭐ after fixing duplication

**Estimated fix time:** 2-3 hours
**Priority:** CRITICAL - Duplication is a serious thesis issue

---

## 7. PROPOSED C3 REORGANIZATION

**Current structure:**
1. c3_chapter.tex (wrapper)
2. c3_approach.tex (4 principles)
3. c3_solution.tex (detailed design)
4. c3_analysis_and_evaluation.tex (mostly duplicate)

**Proposed structure:**
1. c3_chapter.tex (wrapper)
2. c3_approach.tex (keep - unique content)
3. c3_solution.tex (keep - unique content)
4. c3_analysis_and_evaluation.tex → **REPLACE with 5-line summary referencing C2**

Or: Delete c3_analysis_and_evaluation.tex entirely and move its unique content (if any) to c3_solution.tex
