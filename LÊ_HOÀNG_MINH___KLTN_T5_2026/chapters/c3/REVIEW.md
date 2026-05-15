# Chapter 3 Review: Teaching Presentation Development Platform Edlora

**Reviewer:** Nguyễn Thanh Phát  
**Date:** 2026  
**File Reviewed:** c3/c3_chapter.tex (1193 lines)

---

## 1. OVERALL ASSESSMENT

**Strength:** ⭐⭐⭐⭐⭐ (5/5)

Exceptional software engineering documentation. Professional use case specifications (5 UCs), comprehensive AI behavioral rules (7 specifications), detailed multi-agent architecture (12 nodes with full tables), complete database schema, and implementation walkthrough. This chapter serves as a **template for software engineering theses**.

---

## 2. DETAILED SECTION REVIEWS

### 2.1 Requirements Analysis - Use Cases (Lines 6-209)

**Use Case Documentation Excellence:**

**Five Sequential Use Cases:**
1. **UC-01: Authenticate** - Identity verification with Clerk
2. **UC-02: Onboard** - Education level and textbook selection  
3. **UC-03: Manage lessons** - CRUD operations on lesson library
4. **UC-04: Edit lesson** - AI-assisted content creation (core value)
5. **UC-05: Present lesson** - Fullscreen classroom delivery

**Use Case Table Quality:**
All 5 use cases documented in `longtable` format with:
- Use Case ID
- Name
- Actors
- Goal
- Prerequisites
- Trigger
- Success Flow
- Exceptions
- Outcomes

**Critical Strengths:**
- ✅ **Prerequisites chain:** UC-01 → UC-02 → UC-03 → UC-04 → UC-05 (logical dependency)
- ✅ **Actor definitions:** Teacher, Teaching Assistant (AI), Clerk (authentication service)
- ✅ **Exception handling:** Specific error scenarios documented
- ✅ **Preconditions/postconditions:** Clear state requirements

**UC-04 (Edit lesson) - Core Value Proposition:**
```
Goal: Transform partial draft into story-driven experience
Prerequisites: UC-03 complete, user owns lesson
Actors: Teacher, Teaching Assistant (AI)
```

This is the money shot - where the AI actually generates content.

---

### 2.2 AI Specifications (Lines 212-365)

**Outstanding AI Governance Framework:**

**Seven Behavioral Specifications:**

1. **Ground responses in verified inputs** - No hallucination of curriculum/fabricated uploads
2. **Enforce lesson structure contract** - 15-20 segments, 3 required activities (confrontation, vault, boss), no adjacent activities
3. **Keep teaching slides concise** - Max 150 words/1000 chars, describe visuals in words
4. **Require explicit confirmation before building** - Prevent accidental expensive operations
5. **Ground responses in retrieved curriculum evidence** - RAG integration with attribution
6. **Maintain transparency regarding knowledge limits** - Acknowledge missing/partial retrieval
7. **Align to strict precedence hierarchy** - Truthfulness > Structure > Teacher objective > Curriculum > Creative tone

**Example Quality:**
Each spec includes **Compliant** and **Violation** examples with rationale:

```
User: Make the background look like a forest. (no image attached)

Compliant: "I have set the lesson theme to a forest. No background image 
was attached with this message, so presentation backgrounds will stay 
as they were."

Violation: "Thanks for the forest picture you uploaded! I have added it 
behind every slide." (Hallucination)
```

**Precedence Hierarchy (Critical Design Decision):**
1. Truthfulness about evidence (highest)
2. Lesson structure and safety constraints
3. Teacher's current objective and procedure
4. Retrieved official curriculum passages
5. Creative tone and storytelling (lowest)

This prevents the AI from lying to be helpful.

---

### 2.3 System Design - Multi-Agent Architecture (Lines 390-913)

**Exceptional Technical Depth:**

**Architecture Overview:**
- **Root Graph:** Coordinator
- **Lesson Director Subgraph:** Intent comprehension, planning, readiness determination
- **Lesson Developer Subgraph:** Plan → Scene materialization

**12 Active Nodes Documented:**

| Node ID | Agent | Purpose |
|---------|-------|---------|
| `lesson_director` | Root | Entry point, invokes subgraph |
| `lesson_developer` | Root | Invokes developer subgraph |
| `detect_generation_intent` | Director | Decides if user explicitly requests generation |
| `route_director` | Director | Chooses next branch (reply/plan/develop) |
| `extract_lesson_spec` | Director | Extracts objectives and procedures |
| `assess_input_comprehensiveness` | Director | Checks if info sufficient for generation |
| `draft_plan` | Director | Creates canonical creation_plan |
| `delegate_to_developer` | Director | Prepares state envelope for handoff |
| `create_lesson` | Developer | Materializes plan into scenes |
| `report_to_director` | Developer | Summarizes generation outcome |
| `reply_to_user` | Director | Composes final teacher-facing message |
| `finish_turn` | Director | Safe termination with defaults |

**Node Documentation Format:**
Each node table includes:
- ID, Agent (root/director/developer)
- Role (bullet points)
- Execution Trigger (when it runs)
- Recovery/Context & Outcomes (error handling)

**Shared State Documentation (Lines 877-913):**
```
- Conversation history
- Textbook context (RAG retrieved)
- lesson_plan (extracted objectives)
- procedure (teaching phases)
- creation_plan (canonical outline)
- scenes_data (final playable slides)
- generation_status (lifecycle state)
```

---

### 2.4 RAG Architecture (Lines 915-931)

**Three-Phase Pipeline:**
1. **Ingestion:** PDF → Segmentation by headings → Low-temp LLM → Structured chunks
2. **Storage:** Embeddings → PostgreSQL + pgvector
3. **Retrieval:** Similarity search → Textbook context → AI state

**Citation:** `\cite{lewis2020retrievalAugmentedGeneration}`

---

### 2.5 Backend and Database (Lines 933-1084)

**Complete API Endpoint Catalog (Lines 942-967):**

Documented 12 HTTP endpoints with:
- Method, Path, Auth requirements, Role

Examples:
```
POST /api/lessons/create - Bearer required - Creates new lesson with idempotency key
POST /api/lessons/chat - Bearer required - Proxies to AI with SSE streaming
POST /api/storage/upload - Bearer required - Cloudflare R2 multipart upload
```

**ER Diagram (Figure \ref{fig:backend_er_model}):**
- User, Lesson, Scene, Chat History, Textbook Chunk entities

**Database Schema Tables:**

**User Table (Lines 982-1002):**
- id (Clerk identifier, PK)
- email (unique, indexed)
- full_name
- education_level
- textbook_set
- is_active (soft lockout)

**Lesson Table (Lines 1004-1021):**
- id (UUIDv7)
- name
- owner_id (FK to user)
- background_image_url
- background_audio_url

**Scene Table (Lines 1023-1040):**
- id, lesson_id (FK), name, order_index
- content (JSONB - flexible schema for game types)

**Chat History (Lines 1042-1061):**
- Separated from lesson table for performance
- threads (JSONB array of messages)

**Textbook Chunk (Lines 1063-1083):**
- id, textbook_id, unit_name, skill_type
- content (text), embedding (vector(1536))
- IVFFlat index for fast similarity search

**Schema Design Rationale:**
Good explanation at lines 1042 and 1082:
- Chat history separated "ensures querying lesson list remains incredibly fast"
- Textbook chunks decoupled "can serve thousands of teachers simultaneously without needlessly duplicating vector data"

---

### 2.6 Implementation - Use Case Walkthroughs (Lines 1085-1193)

**Complete UI + Sequence Diagrams for All 5 Use Cases:**

**Format for each UC:**
- Placeholder screenshot (UI mockup)
- Placeholder sequence diagram (system interaction)
- Detailed prose explanation

**Example: UC-04 Lesson Creation (Lines 1153-1173):**
- UI: Three-column layout (scenes, preview, chat)
- Sequence: GET lesson → POST chat with SSE → RAG retrieval → AI streaming → PATCH updates
- Technical details: Server-Sent Events, progressive streaming

**Notable Features:**
- Idempotency keys for lesson creation (prevents duplicates)
- Clerk webhooks for lifecycle events
- Cloudflare R2 for multimedia storage
- Optimistic UI updates mentioned

---

## 3. CROSS-CUTTING ISSUES

### 3.1 Figure References

| Figure | Reference | Status |
|--------|-----------|--------|
| Use case diagram | `\ref{fig:use_case_diagram}` | ✅ |
| High-level architecture | `\ref{fig:high_level_architecture}` | ⚠️ Line 383: Image commented out |
| Root graph | `\ref{fig:multi_agent_root_graph}` | ✅ |
| Lesson Director | `\ref{fig:lesson_director_mas_architecture}` | ✅ |
| Lesson Developer | `\ref{fig:lesson_developer_mas_architecture}` | ✅ |
| RAG design | `\ref{fig:rag_design_architecture}` | ✅ |
| ER diagram | `\ref{fig:backend_er_model}` | ✅ |

**Critical Issue:** Line 383 has high-level architecture figure **commented out**:
```latex
% \includegraphics[width=1\textwidth]{figures/high_level_architecture.png}
```

### 3.2 Table References

**All tables properly labeled:**
- 5 use case tables (UC-01 through UC-05)
- 12 multi-agent node tables
- 1 shared state table
- 5 database schema tables

**Consistent formatting:** Using `longtable` with proper headers/footers for multi-page

### 3.3 Implementation Placeholders

**Lines 1095-1193:** All implementation subsections have placeholder figures:
```latex
\fbox{\parbox{0.92\textwidth}{\centering Placeholder screenshot: ...}}
```

**Status:** 
- ⚠️ These need to be replaced with actual screenshots/diagrams before defense
- But placeholders are clearly marked and won't break compilation

---

## 4. RECOMMENDATIONS

### Critical fixes:
1. [ ] **Uncomment or remove** high-level architecture figure (line 383)
2. [ ] Replace all placeholder screenshots with actual figures
3. [ ] Replace all placeholder sequence diagrams with actual diagrams

### Improvements:
1. [ ] Add cost estimate for AI operations (token usage per lesson generation)
2. [ ] Add latency benchmarks for each use case
3. [ ] Consider adding error rate metrics from testing

### Questions for author:
1. Do the placeholder screenshots need to be captured from production or staging?
2. Are the sequence diagrams available in engineering docs to be redrawn?
3. What is the actual token cost per lesson generation?

---

## 5. COMPARISON WITH PEER THESES

| Aspect | Minh's C3 | Quân's C3 | My C3 (Phat) |
|--------|-----------|-----------|--------------|
| **Length** | 1193 lines | ~250 lines | 307 lines |
| **Use Cases** | 5 detailed specs | None | Minimal |
| **AI Rules** | 7 behavioral specs | Prompt engineering | Architecture |
| **Architecture** | 12-node multi-agent | Self-healing loop | Dual-pipeline NN |
| **Database** | Complete schema | Minimal | None |
| **Implementation** | 5 UC walkthroughs | Architecture | Pseudocode |

**Unique Strengths of Minh's C3:**
1. Professional use case specifications
2. AI governance framework (precedence hierarchy)
3. Complete node-level multi-agent documentation
4. Full database schema with design rationale
5. Implementation walkthroughs with UI+sequence diagrams

---

## 6. FINAL VERDICT

Chapter 3 is **publication-quality software engineering documentation**.

**Strengths:**
- ✅ Enterprise-grade use case analysis
- ✅ Ethical AI governance (truthfulness constraints)
- ✅ Complete system architecture
- ✅ Production-ready database design
- ✅ Implementation walkthrough

**Issues:**
- High-level architecture figure commented out
- Implementation placeholders need actual screenshots

**Estimated fix time:** 3-4 hours (mainly capturing screenshots)

**Rating:** ⭐⭐⭐⭐⭐ (5/5) - Once placeholders are filled, this is thesis gold

---

## 7. WHAT I CAN LEARN FOR MY THESIS

**Use Case Documentation:**
- Template: ID, Name, Actors, Goal, Prerequisites, Trigger, Flow, Exceptions, Outcomes
- Chain prerequisites to show logical dependencies

**AI Governance:**
- Precedence hierarchy prevents harmful helpfulness
- Concrete compliant/violation examples for each rule

**Database Design:**
- Separation of concerns (chat history separate from lessons)
- JSONB for flexible schema evolution
- Design rationale explanations

**Implementation Section:**
- UI screenshot + sequence diagram + prose for each use case
- Clear technical details (idempotency keys, SSE, webhooks)
