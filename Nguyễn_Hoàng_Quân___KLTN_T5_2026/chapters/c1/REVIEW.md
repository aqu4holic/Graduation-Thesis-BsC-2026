# Chapter 1 Review: Đặt vấn đề

**Reviewer:** Nguyễn Thanh Phát  
**Date:** 2026  
**Files Reviewed:** c1_introduction.tex, c1_context.tex, c1_problem.tex, c1_purpose.tex

---

## 1. OVERALL ASSESSMENT

**Strength:** ⭐⭐⭐⭐☆ (4/5)

Chapter 1 presents a clear problem statement with good logical flow from context → problem → solution approach. The writing is technically sound in Vietnamese.

---

## 2. DETAILED SECTION REVIEWS

### 2.1 c1_context.tex - Giới thiệu về bài toán

**Strengths:**
- ✅ Clear introduction to CI/CD context and unit testing importance
- ✅ Good citation usage (`\cite{unittesing, unittesting1}`)
- ✅ Introduces Qodo Cover tool appropriately
- ✅ Explains the core problem: self-healing optimization for LLM-generated tests

**Suggestions for improvement:**
1. **Add a hook:** Consider starting with a concrete example or statistic about test generation failure rates to grab attention
2. **Clarify timeline:** "15/06/2025" appears later in problem.tex as the Qodo Cover discontinuation date - make this consistent if referenced here
3. **Expand acronyms:** First mention of CI/CD should spell out "Continuous Integration/Continuous Deployment"

**Writing quality:** Good technical Vietnamese, appropriate formal tone

---

### 2.2 c1_problem.tex - Những thách thức lớn trong bài toán

**Strengths:**
- ✅ Well-structured with 3 clear challenges:
  1. CLI limitation (no IDE integration)
  2. Resource waste from test discarding
  3. LLM instability with large context windows
- ✅ Good technical analysis of why batch-processing fails
- ✅ Cites relevant literature (hallucination, context window issues)
- ✅ Specific technical terms: "không gian tên" (namespace), "ảo giác" (hallucination)

**Critical Issues:**
1. **Redundancy:** The discontinuation date "15/06/2025" and CLI limitation appear here AND in c3_analysis_and_evaluation.tex (similar text). Consider which location is more appropriate.

**Suggestions:**
1. **Quantify impact:** Add statistics if available (e.g., "40% of generated tests fail due to syntax errors")
2. **Diagram reference:** The text references Hình \ref{fig:batch_processing} but it's in c3_analysis_and_evaluation.tex - ensure figures are in logical order
3. **Simplify sentence structure:** Some sentences are very long (e.g., line 3 is 5+ clauses). Break into shorter sentences for readability.

**Example of complex sentence (line 3):**
> "Thách thức đầu tiên và rõ nét nhất xuất phát từ rào cản nền tảng của các công cụ tiền nhiệm..."

This could be split into 2-3 sentences.

---

### 2.3 c1_purpose.tex - Hướng tiếp cận và đóng góp

**Strengths:**
- ✅ Clear system diagram reference (Fig \ref{fig:intro})
- ✅ Well-structured contributions list (4 bullet points)
- ✅ Quantified results mentioned: "5-36% line coverage, 10-47% branch coverage"
- ✅ Good explanation of the self-healing feedback loop
- ✅ Clear chapter roadmap at the end

**Critical Issues:**
1. **Discrepancy:** Line 10 says Qodo Cover stopped development "15/06/2025" but this already appeared in c1_problem.tex
2. **Figure label:** Line 6 references `intro.png` but the caption is generic ("Sơ đồ hệ thống Qodo Plus") - make more descriptive

**Suggestions:**
1. **Move roadmap to introduction.tex:** The chapter outline at the end (lines 22-28) duplicates what's likely in c1_introduction.tex - verify and consolidate
2. **Add concrete metrics:** "max-fix-attempts" mentioned - specify the actual number used (e.g., 3 attempts?)
3. **Strengthen contribution claims:**
   - "Xây dựng nền tảng sửa lỗi cục bộ tự động" → Good
   - "Cải thiện tính ổn định" → How measured?
   - "Tối ưu hóa tài nguyên" → Specific token savings?

**Citation issue:** Line 10 has `\cite{qodo}` - ensure this is in references.bib

---

## 3. CROSS-CUTTING ISSUES

### 3.1 Content Duplication
The following content appears in multiple files:
- Qodo Cover discontinuation date (15/06/2025)
- CLI limitation explanation
- Batch-processing problem description

**Recommendation:** Keep in c1_problem.tex (most logical place) and reference briefly in c3.

### 3.2 Figure References
- \ref{fig:intro} - in c1_purpose.tex (appropriate)
- \ref{fig:batch_processing} - referenced in c1_problem.tex but defined in c3_analysis_and_evaluation.tex

**Fix:** Either move figure to Chapter 1, or remove reference until Chapter 3.

### 3.3 Citation Consistency
- `\cite{qodo}` appears twice - verify in .bib file
- `\cite{fan2023automatedrepairprogramslarge}` in c1_purpose - good recent citation

---

## 4. RECOMMENDATIONS

### Immediate fixes:
1. [ ] Remove duplicate discontinuation date from c1_purpose.tex
2. [ ] Fix figure reference \ref{fig:batch_processing} in c1_problem.tex
3. [ ] Verify all citations exist in references.bib

### Improvements:
1. [ ] Add attention-grabbing statistic to c1_context.tex opening
2. [ ] Break up long sentences in c1_problem.tex
3. [ ] Add specific token/cost savings metrics to contribution list
4. [ ] Consider merging c1_context.tex and c1_problem.tex (both are relatively short)

### Questions for author:
1. Is the 15/06/2025 date confirmed or estimated?
2. What is the actual max_fix_attempts value used?
3. Are there any quantitative metrics on test generation failure rates?

---

## 5. COMPARISON WITH MY THESIS (PHAT)

| Aspect | Quân's C1 | My C1 | Notes |
|--------|-----------|-------|-------|
| **Structure** | Context→Problem→Purpose | Problem→Motivation→Contributions | Both logical |
| **Length** | ~35 lines across 3 files | 214 lines single file | Quân more modular |
| **Quantification** | Coverage % mentioned | Multiple % metrics | Both data-driven |
| **Figures** | 1 figure | Multiple figures | |
| **Citations** | ~10 citations | ~15 citations | Good density |

**What I can learn:**
- Modular sub-file approach is clean and maintainable
- Clear 3-challenge structure in problem statement
- Good use of Vietnamese technical terminology

---

## 6. FINAL VERDICT

Chapter 1 effectively establishes the research problem. The main issues are:
1. Content duplication across files
2. Some long, complex sentences
3. Missing specific metrics in contribution claims

**Estimated time to fix:** 1-2 hours

**Priority fixes:** Content duplication, figure reference
