# Chapter 2 Review: Background and Related Work

**Reviewer:** Nguyễn Thanh Phát  
**Date:** 2026  
**File Reviewed:** c2/c2_chapter.tex (152 lines, but very dense)

---

## 1. OVERALL ASSESSMENT

**Strength:** ⭐⭐⭐⭐⭐ (5/5)

Outstanding theoretical foundation combining cognitive science, pedagogy, and comprehensive technical architecture. Exceptional depth in storytelling research with quantified retention studies. Complete technology stack documentation from AI to frontend to backend.

---

## 2. DETAILED SECTION REVIEWS

### 2.1 Teaching Pedagogies - Storytelling (Lines 15-35)

**Research Depth Excellence:**

**Three Key Studies Cited:**

1. **Schank & Abelson (1995)** - "Knowledge and Memory: The Real Story"
   - Foundation: Human memory organized as interconnected stories
   - Key insight: "The act of telling is synonymous with the act of remembering"
   - Application: Narrative as cognitive scaffold

2. **Graeber, Roth & Zimmermann (2024)** - "Stories, Statistics, and Memory"
   - **Quantified results:**
     - Statistical data retention decay: 73% after 24 hours
     - Narrative retention decay: only 32% after 24 hours
     - Narrative recall accuracy: 61.6% vs 28.7% for statistics
   - **Critical constraint for Edlora:** Abstract data must be contextualized

3. **Bower & Clark (1969)** - "Narrative stories as mediators for serial learning"
   - **Results:**
     - Narrative group recall: 93% median
     - Control (rote memorization): 13% recall
     - Lowest narrative performer > highest control performer

**Analysis Quality:**
- ✅ Specific, quantified results from peer-reviewed sources
- ✅ Clear application to Edlora architecture
- ✅ Synthesizes findings into design principles
- ✅ Evolutionary/historical context (Paleolithic era)

**Technical Highlight:**
The paragraph at lines 21-23 explains the cognitive mechanism beautifully: "When a student encounters a discrete grammatical rule... the brain struggles to encode it because it lacks an associative index. However, when that rule is embedded within a narrative... the brain utilizes the narrative arc as a cognitive scaffold."

---

### 2.2 Teaching Pedagogies - Gamification (Lines 37-51)

**Research Quality:**

**Key Studies:**

1. **Lister (2015)** - Gamification effects on motivation
   - 12.9% reduction in course failure rates
   - Transition from passive consumption to active participation
   - Dopamine release in reward pathways

2. **Hamari et al. (2014)** - "Does Gamification Work?"
   - Positive impacts validated
   - **Critical caveat:** Novelty effect - extrinsic rewards decay over time
   - Must be intrinsically motivating

3. **Jihadillah (2025)** - Concept-binding principle
   - Game mechanics must be tightly coupled with academic concepts
   - Superficial scoring (multiple-choice + points) inferior to mechanics representing actual concepts

**Edlora's Approach:**
- ✅ Addresses novelty effect by making narrative = core academic concept
- ✅ Games advance the story (concept-binding achieved)
- ✅ Gamification as modular option (preserves teacher autonomy)

---

### 2.3 AI Technologies - RAG (Lines 59-73)

**Technical Depth:**

**Cosine Similarity Equation (\ref{eq:cosine}):**
```
Cosine Similarity = (A·B) / (||A|| ||B||)
```

**Implementation Stack:**
- MinerU - PDF parsing with reading order preservation
- OpenAI Embeddings - semantic translation
- PostgreSQL + pgvector extension - vector storage and similarity search

**Quality:**
- ✅ Mathematical foundation explained
- ✅ Specific tool choices justified
- ✅ Curriculum grounding (Vietnam's Ministry of Education)

---

### 2.4 AI Technologies - Agentic AI (Lines 75-81)

**Multi-Agent Architecture:**
- LangGraph for stateful routing (DAG structure)
- LangChain for tool standardization
- Persistent memory across generation cycles

**Citation:** ReAct (Yao et al. 2022) - `\cite{yao2022react}`

---

### 2.5 AI Technologies - LLMs (Lines 83-89)

**Model Selection:**
- **GPT 5.4** - Primary creative engine (production)
- **GPT 5.4 Nano** - Cost-effective variant (development/testing)
- **Testing:** pytest, LangSmith, OpenEvals
- **Deployment:** Docker container on Render

**Note:** "GPT 5.4" appears - verify if this is correct (should be GPT-4o?)

---

### 2.6 Frontend Technologies (Lines 90-115)

**Exceptional Technical Detail:**

**Rendering:**
- Virtual DOM paradigm explained
- TypeScript + React + Next.js

**State Management:**
- Zustand (client state)
- TanStack React Query (server state)
- React Hook Form + Zod (validation)

**Editing Interface:**
- TipTap (headless block-based editor)
- Tailwind CSS + MUI (Material Design for Google ecosystem familiarity)

**Testing:**
- vitest, Storybook, Playwright, Chromatic
- Docker container on Vercel

**Design Rationale:**
Good justification for MUI - "educators spend most time in Google Workspace" (line 108)

---

### 2.7 Backend Technologies (Lines 117-139)

**WSGI vs ASGI Explanation (Excellent):**

**Figure referenced:** \ref{fig:wsgi_architecture} and \ref{fig:asgi_architecture}

**Key Insight:**
- WSGI: Dedicated worker thread per request → blocks on slow I/O (AI generation)
- ASGI: Non-blocking event loop → handles concurrent requests efficiently

**Technology Stack:**
- FastAPI + Uvicorn (async)
- Pydantic (data validation)
- PostgreSQL + SQLModel + Alembic
- Clerk (authentication)
- Cloudflare R2 (object storage, zero egress cost)
- Supabase (managed PostgreSQL)

**Economic Consideration:**
"capitalizing on its zero-cost egress traffic model" (line 134) - good practical detail

---

### 2.8 CI/CD (Lines 141-151)

**DevOps Maturity:**

**GitHub Actions Workflows:**
- Lint: ESLint, TypeScript compiler, Ruff, Black
- Tests: vitest, pytest
- Visual: Chromatic
- Release: standard-versioning

**Git Hooks (Husky):**
- lint-staged
- commitlint
- Prettier

**Note:** "certain gaps exist...omitting complex Playwright E2E tests" - honest limitation

---

## 3. CROSS-CUTTING ISSUES

### 3.1 Citation Quality

**Cognitive Science/Pedagogy:**
- Schank & Abelson (1995) - classic, foundational
- Graeber et al. (2024) - recent, quantified
- Bower & Clark (1969) - classic experimental study
- Hamari et al. (2014) - comprehensive review
- Lister (2015) - specific effect sizes
- Jihadillah (2025) - very recent

**Technical:**
- Lewis et al. (2020) - RAG foundational
- Yao et al. (2022) - ReAct agent framework
- Vaswani et al. (2017) - Transformer (implied, not explicitly cited)

**Good mix:** 40% classic/60% recent, showing both foundational understanding and current awareness

### 3.2 Technical Architecture Documentation

| Layer | Components | Coverage |
|-------|-----------|----------|
| AI/ML | RAG, Agentic AI, LLMs | ⭐⭐⭐⭐⭐ |
| Frontend | Rendering, State, UI, Testing | ⭐⭐⭐⭐⭐ |
| Backend | API, Database, Auth, Storage | ⭐⭐⭐⭐⭐ |
| DevOps | CI/CD, Testing, Deployment | ⭐⭐⭐⭐⭐ |

**Most comprehensive technical chapter among all three theses**

### 3.3 Equations and Figures

| Item | Reference | Status |
|------|-----------|--------|
| Cosine Similarity | `\ref{eq:cosine}` | ✅ |
| WSGI Architecture | `\ref{fig:wsgi_architecture}` | ⚠️ Verify |
| ASGI Architecture | `\ref{fig:asgi_architecture}` | ⚠️ Verify |

---

## 4. RECOMMENDATIONS

### Immediate fixes:
1. [ ] **Verify model name:** "GPT 5.4" (line 86) - likely should be GPT-4o
2. [ ] Check WSGI/ASGI architecture figures exist and render clearly
3. [ ] Verify all citations in references.bib

### Improvements:
1. [ ] Consider adding simplified architecture diagram showing all layers together
2. [ ] Could add cost analysis table (R2 egress savings, Supabase vs self-hosted)
3. [ ] Add brief comparison table of selected tools vs. alternatives

### Questions for author:
1. Is "GPT 5.4" the correct model name or should it be GPT-4o?
2. Are the WSGI/ASGI diagrams clear enough for non-technical readers?
3. Why Render for backend but Vercel for frontend? (different platforms)

---

## 5. COMPARISON WITH PEER THESES

| Aspect | Minh's C2 | Quân's C2 | My C2 (Phat) |
|--------|-----------|-----------|--------------|
| **Pedagogy** | Extensive (storytelling + gamification) | None (pure technical) | Minimal |
| **Cognitive Science** | Deep with quantified results | None | None |
| **Technical Stack** | Complete (AI→Frontend→Backend→DevOps) | Testing-focused | ML-focused |
| **Equations** | 1 (cosine similarity) | None | Multiple |
| **Architecture** | Full system | pytest + self-healing | Neural nets |
| **Citations** | 15+ (diverse) | 20+ (testing/AI) | 20+ (causal ML) |

**What I can learn:**
1. **Quantified cognitive science** - retention rates, failure rate reductions
2. **Complete stack documentation** - from AI to DevOps
3. **Economic justifications** - zero egress, cost optimization
4. **Architecture explanations** - WSGI vs ASGI clearly explained

---

## 6. FINAL VERDICT

Chapter 2 establishes **exceptional theoretical and technical foundations**.

**Unique Strengths:**
1. ✅ **Pedagogy-first approach** - cognitive science before technology
2. ✅ **Quantified research** - specific retention/failure rate numbers
3. ✅ **Full-stack coverage** - no gaps from AI to deployment
4. ✅ **Economic awareness** - cost considerations throughout

**Minor Issues:**
- Model name "GPT 5.4" likely incorrect
- Figure references need verification

**Estimated fix time:** 1 hour

**Rating:** ⭐⭐⭐⭐⭐ (5/5) - Publication-quality technical documentation

---

## 7. KEY TAKEAWAYS FOR MY THESIS

**Storytelling Evidence:**
- 73% vs 32% retention decay (24 hours)
- 61.6% vs 28.7% recall accuracy
- 93% vs 13% serial learning recall

**Architecture Pattern:**
- Pedagogy → AI → Frontend → Backend → DevOps
- Each layer justified with specific tools and economic rationale

**Writing Quality:**
- Technical concepts explained for educated non-specialists
- Bold text for key terms
- Clear transitions between sections
