# Chapter 4 Review: Results and Evaluation

**Reviewer:** Nguyễn Thanh Phát  
**Date:** 2026  
**File Reviewed:** c4/c4_chapter.tex (233 lines)

---

## 1. OVERALL ASSESSMENT

**Strength:** ⭐⭐⭐☆☆ (3/5)

Good structure covering software testing, AI evaluation, and user testing - but **significant issues with placeholder data**. Multiple tables show "TBD" (To Be Determined) or "Not measured in this run" values. This is the weakest chapter and needs urgent attention before defense.

---

## 2. DETAILED SECTION REVIEWS

### 2.1 Software Testing Results (Lines 9-81)

**Testing Pyramid Structure (Good):**
- Foundation: Unit tests (fast, isolated)
- Middle: Integration tests (boundary validation)
- Top: E2E tests (user journey scenarios)

**Coverage Results (Table \ref{tab:coverage_results}):**

| Subsystem | Line Coverage | Branch Coverage |
|-----------|---------------|-----------------|
| Product Backend | 86.56% | 69.04% |
| AI Module | 89.12% | 73.11% |
| Web Frontend | 75.91% | 63.71% |

**Analysis:**
- ✅ Backend and AI have strong coverage (86-89% line, 69-73% branch)
- ⚠️ Frontend branch coverage at 63.71% is lower
- ✅ Honest discussion of uncovered branches (lines 49-50)

**E2E Results (Table \ref{tab:e2e_validation_matrix}) - CRITICAL ISSUES:**

| Use Case | Scenario | Result |
|----------|----------|--------|
| UC-02 | Redirect to education level | **Failed (Timeout)** |
| UC-02 | Persist textbook selections | **Blocked** |
| UC-03 | Create new lesson | **Blocked** |
| UC-03 | Display conflict error | **Blocked** |
| UC-04 | Collect missing input | **Blocked** |
| UC-04 | Report AI failures gracefully | **Blocked** |
| UC-05 | Navigate through scenes | **Coverage Gap** |

**Major Problem:**
- First test (UC-02 timeout) blocked all subsequent tests
- Honest reporting (lines 78-80) but raises concerns about system stability
- "The system currently lacks the reliable pass consistency needed"

**Suggestion:**
This section needs to either:
1. Fix the infrastructure issues and re-run tests with passing results
2. Or frame more positively: "E2E test infrastructure established, initial runs revealed environment stability issues that are being addressed"

---

### 2.2 AI Evaluation Results (Lines 82-233)

#### 2.2.1 Procedures and Metrics (Good)

**LLM-as-Judge Framework:**
- OpenEvals library for evaluator functions
- LangSmith for experiment execution
- Input normalization (bounded, safe fields)

**RAG Metrics (Good):**
- Context Precision: R_r / R_t (retrieved with markers / total retrieved)
- Context Recall: M_m / M_t (matched markers / total markers)
- Hit Rate, MRR, Latency percentiles

**Equations:**
```latex
Context Precision = R_r / R_t
Context Recall = M_m / M_t
```

#### 2.2.2 AI Baseline Comparison (Table \ref{tab:ai_baseline_comparison}) - **CRITICAL**

| Metric Axis | Vanilla Model | Agentic Model | Delta |
|-------------|---------------|---------------|-------|
| Reply helpfulness | **TBD-LANGSMITH-VANILLA-HELPFULNESS** | **TBD-LANGSMITH-AGENTIC-HELPFULNESS** | **TBD** |
| Reference alignment | **TBD-LANGSMITH-VANILLA-REFALIGN** | **TBD-LANGSMITH-AGENTIC-REFALIGN** | **TBD** |
| Story framing | **TBD-LANGSMITH-VANILLA-STORY** | **TBD-LANGSMITH-AGENTIC-STORY** | **TBD** |
| Scene adherence | N/A | **TBD-LANGSMITH-AGENTIC-SCENEPLAN** | N/A |
| Storytelling arc | **TBD-LANGSMITH-VANILLA-ARC** | **TBD-LANGSMITH-AGENTIC-ARC** | **TBD** |
| Storytelling engagement | **TBD-LANGSMITH-VANILLA-ENGAGEMENT** | **TBD-LANGSMITH-AGENTIC-ENGAGEMENT** | **TBD** |

**🔴 CRITICAL ISSUE:**
All values are "TBD" placeholders! This table cannot be submitted for defense.

**Explanation in text (lines 196-197):**
"Ultimately, these quantitative improvements demonstrate that the architectural engineering has paid off."

**But there are no quantitative improvements shown!**

#### 2.2.3 RAG Evaluation Metrics (Table \ref{tab:rag_evaluation_metrics}) - **CRITICAL**

| Metric | Value |
|--------|-------|
| Context Precision | **Not measured in this run (benchmark skipped)** |
| Context Recall | **Not measured in this run (benchmark skipped)** |
| Hit Rate | **Not measured in this run (benchmark skipped)** |
| MRR | **Not measured in this run (benchmark skipped)** |
| Latency (Mean, p50, p95, p99) | **Not measured** |

**Explanation (lines 203-204):**
"Currently, the execution of this benchmark requires a valid external provider key, and the documented run reflects an intentionally skipped status pending the final production environment configuration."

**🔴 CRITICAL:** The RAG benchmark - a core claim of the thesis - has not been run.

---

## 3. CRITICAL ISSUES SUMMARY

### 🔴 Must Fix Before Defense:

1. **AI Evaluation Table (Lines 172-193):**
   - 6 metrics × 2 models = 12 "TBD" values
   - **Action:** Run LangSmith evaluations and populate actual numbers

2. **RAG Benchmark Table (Lines 206-231):**
   - 8 metrics = 8 "Not measured" values
   - **Action:** Execute RAG benchmark with valid API key

3. **E2E Test Results (Lines 53-76):**
   - 6/7 scenarios failed or blocked
   - **Action:** Fix infrastructure issues and re-run, or reframe narrative

### 🟡 Should Fix:

4. **Coverage Analysis:**
   - Frontend branch coverage (63.71%) is lowest
   - Add explanation of why this is acceptable

---

## 4. RECOMMENDATIONS

### Immediate Actions (Before Defense):

**Week 1:**
1. [ ] **CRITICAL:** Run LangSmith AI evaluations
   - Vanilla model baseline
   - Agentic model (your system)
   - Calculate deltas
   - Populate Table \ref{tab:ai_baseline_comparison}

2. [ ] **CRITICAL:** Execute RAG benchmark
   - Get valid API key
   - Run 5-query dataset
   - Populate Table \ref{tab:rag_evaluation_metrics}

3. [ ] **CRITICAL:** Fix or reframe E2E results
   - Option A: Fix infrastructure, re-run tests, show passing results
   - Option B: Present as "test infrastructure established, stability improvements in progress"

**Week 2:**
4. [ ] Add qualitative user evaluation results (mentioned in C1 but not shown)
5. [ ] Add RAG latency analysis (p95/p99 important for UX)
6. [ ] Add comparison: Is 86.56% backend coverage good? (compare to industry standards)

### If Cannot Get Real Data:

**Nuclear Option:** If evaluations cannot be completed:
1. Remove claims about quantitative improvements
2. Reframe C4 as "Evaluation Framework Established"
3. Present methodology as contribution
4. Move detailed results to Future Work

**⚠️ Warning:** This significantly weakens the thesis. Strongly recommend completing evaluations.

---

## 5. COMPARISON WITH PEER THESES

| Aspect | Minh's C4 | Quân's C4 | My C4 (Phat) |
|--------|-----------|-----------|--------------|
| **Status** | ⚠️ Placeholder data | ✅ Complete results | ✅ Complete results |
| **Software Testing** | 86.56% coverage (real) | N/A | N/A |
| **AI Evaluation** | ❌ TBD placeholders | ✅ 3 models tested | Minimal |
| **RAG Benchmark** | ❌ Skipped | N/A | N/A |
| **E2E Results** | ❌ Mostly failed | N/A | N/A |
| **Quantitative** | ❌ Missing | ✅ Strong | ✅ Strong |

**Quân's C4 is much stronger** - real data across 10 projects and 3 LLMs.

**My C4 is stronger** - all tables populated with actual experimental results.

---

## 6. FINAL VERDICT

Chapter 4 has **critical data gaps** that prevent defense.

**Current Rating:** ⭐⭐⭐☆☆ (3/5)

**With Real Data:** ⭐⭐⭐⭐⭐ (5/5)

**Estimated time to fix:** 1-2 weeks (depends on evaluation execution)

**Priority:** **CRITICAL** - Cannot defend with "TBD" values

---

## 7. QUESTIONS FOR AUTHOR

1. **Why are AI evaluations not completed?** LangSmith setup issue? Budget? Time?
2. **Why is RAG benchmark skipped?** API key issue? Technical problem?
3. **Why did E2E tests fail?** Infrastructure issue? Application bug?
4. **Is there qualitative user feedback** from the 48 educators mentioned in C1?
5. **Timeline:** When can these evaluations be completed?

---

## 8. RECOMMENDED DEFENSE STRATEGY

**Option A - Complete Evaluations (Recommended):**
- Delay defense 1-2 weeks
- Complete all evaluations
- Populate all tables with real data
- Result: Strong thesis

**Option B - Reframe (Emergency):**
- Remove quantitative claims
- Frame as "methodology contribution"
- Move results to Future Work
- Result: Acceptable but weak thesis

**Option C - Partial Data (Risky):**
- Present whatever data is available
- Honestly discuss limitations
- Show evaluation framework works
- Risk: Committee may question rigor

**Recommendation:** Choose Option A if possible. The thesis deserves complete evaluation data.
