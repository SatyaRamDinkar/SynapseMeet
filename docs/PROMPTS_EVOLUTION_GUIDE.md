# Prompt Engineering & Evolution Guide
## Systematic Incremental Prompts for Developing and Extending SynapseMeet

This guide documents the prompt sequences used to design, benchmark, and extend **SynapseMeet** beyond the ACM SAC 2025 base paper (*Bhuvaji et al.*). Use these prompts to reproduce the system, fine-tune models, or drive future voice-assisted iterations.

---

### Phase 1: Ingestion and Diarization Prompts
```text
PROMPT 1.1: Speaker Attribution & Structured Utterance Extraction
You are an expert audio transcription parser. Given raw multi-speaker transcript segments:
1. Normalize speaker tags into canonical speaker identifiers.
2. Group consecutive utterances by speaker and associate start and end timestamps.
3. Extract core conversational turns into the Pydantic schema:
   SpeakerUtterance(speaker_id: str, start_time: float, end_time: float, text: str).
Return exclusively valid JSON matching the schema.
```

---

### Phase 2: Meeting Summary Clustering Prompts (Base Paper Integration)
```text
PROMPT 2.1: Domain-Specific Meeting Summary Synthesis
Analyze the following multi-speaker meeting transcript. Generate:
1. A concise 3-sentence executive summary focusing strictly on architectural choices, technical policies, deadlines, and ownership assignments.
2. Identify the primary technical domain (e.g. Infrastructure, Client Applications, DevOps, Security).
3. Produce key indexing tokens for Agglomerative TF-IDF clustering to group this meeting with semantically related past meetings.
```

---

### Phase 3: Deterministic Decision & Action Item Extraction Prompts
```text
PROMPT 3.1: Strict Action Item & Decision Item Parsing
Extract all firm decisions and commitments from the meeting transcript below.
Rules:
- Do not extract speculative discussions or exploratory brainstorming.
- Only extract a DecisionItem when there is explicit consensus or an authoritative owner stance.
- Map action items to specific assignees and extract explicit or implicit due dates.
Output format:
{
  "decisions": [
    {
      "decision_id": "DEC_<hash>",
      "topic": "<domain>",
      "statement": "<exact verified decision>",
      "owner": "<name>",
      "confidence": <0.0 to 1.0>
    }
  ],
  "action_items": [
    {
      "task_id": "TASK_<hash>",
      "description": "<task>",
      "assignee": "<name>",
      "deadline": "<ISO 8601 or null>"
    }
  ]
}
```

---

### Phase 4: Hybrid RAG & Memory Prompts
```text
PROMPT 4.1: Temporal Cross-Meeting Retrieval Query Generation
Given a newly proposed decision:
Statement: "{new_decision_statement}"
Topic: "{topic}"
Generate:
1. Lexical BM25 search keywords (focusing on system names, database engines, vendor names, protocol versions).
2. Semantic query expansion for dense cosine similarity retrieval against the historical meeting vector store.
Format as JSON: {"lexical_tokens": [...], "dense_expansion": "..."}
```

---

### Phase 5: NLI Contradiction & Conflict Resolution Prompts
```text
PROMPT 5.1: Natural Language Inference (NLI) Contradiction Scoring
You are an expert NLI Contradiction Judge.
Premise (Historical Decision from past meeting): "{past_decision_statement}" (Meeting: {past_meeting_id})
Hypothesis (New Proposed Decision from current meeting): "{new_decision_statement}" (Meeting: {new_meeting_id})

Evaluate whether the Hypothesis contradicts, entails, or is neutral relative to the Premise.
Provide:
1. Entailment probability: float [0.0 - 1.0]
2. Neutral probability: float [0.0 - 1.0]
3. Contradiction probability: float [0.0 - 1.0]
4. Diagnostic Root Cause Explanation: Explain the concrete conflict (e.g. architecture reversal, unaligned deadlines, policy drift).
If Contradiction > 0.70, flag: "REQUIRES_HUMAN_OVERRIDE: TRUE".
```

---

### Phase 6: Human Confirmation & State Machine Prompts
```text
PROMPT 6.1: Conflict Alert Card Synthesis
A contradiction has been detected between past meeting "{past_meeting_id}" and current meeting "{new_meeting_id}".
Synthesize a concise conflict card for the Engineering Lead containing:
- Domain Topic
- Previous Approved Decision & Date
- Proposed New Decision & Proposer
- Risk Assessment of Overriding
- Recommended Actions: [1] Override & Supersede Previous Decision, [2] Reject New Proposal, [3] Schedule Alignment Sync.
```
