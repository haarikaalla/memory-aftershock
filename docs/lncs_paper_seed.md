# Memory Aftershock: Budgeted Repair of Agent Memories under Incomplete Provenance

## LNCS-style paper seed

### Abstract

Long-term AI agents increasingly store user preferences, plans, and derived summaries across sessions. Existing memory benchmarks largely evaluate whether an agent can retrieve the correct evidence, but a corrected memory can leave a second-order failure: downstream memories created from the old fact may remain active. We introduce Memory Aftershock, an end-to-end framework for measuring and repairing these downstream inconsistencies under limited verification budgets. The system combines leakage-safe session retrieval on LongMemEval-cleaned with a controlled aftershock benchmark where dependency ground truth is hidden from the repair policy. A learned dependency estimator ranks candidate downstream memories when explicit provenance is incomplete, and an append-only memory store records corrections, quarantines, replacements, and audit traces. On 470 scored LongMemEval-cleaned questions, the retriever reaches hit@5 of 0.9404 and full-evidence@5 of 0.7809. On the synthetic aftershock benchmark with budget 2, learned dependency repair reaches 1.0000 recall while recorded provenance reaches 0.6607 recall under missing edges. These results suggest that memory repair should be evaluated as a budgeted downstream maintenance problem rather than only as source retrieval or single-fact editing.

### Keywords

Agent memory, memory repair, retrieval-augmented agents, provenance, LongMemEval, graph repair, AI safety evaluation.

### Paper structure

1. Introduction: why downstream memory damage matters for persistent agents.
2. Related work: long-context memory benchmarks, memory editing, rollback repair, agent state maintenance.
3. Problem definition: root correction, downstream dependency, incomplete provenance, verification budget.
4. System: retrieval, append-only memory store, dependency estimator, repair engine, audit trace.
5. Benchmarks: LongMemEval-cleaned retrieval and synthetic aftershock repair.
6. Results: retrieval metrics, repair metrics, error analysis.
7. Limitations: synthetic verifier, simple dependency model, no LLM-based real-world verifier yet.
8. Conclusion: aftershock repair as a core requirement for reliable long-term agents.
