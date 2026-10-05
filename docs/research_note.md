# Research Note

Memory Aftershock studies a narrow failure mode in long-term AI agents: a single corrected memory can leave stale plans, recommendations, and summaries downstream. The project treats memory repair as a budgeted graph problem. The system must decide which downstream memories to verify when provenance is incomplete and verification calls are limited.

The benchmark separates two claims:

1. LongMemEval retrieval measures whether the retriever can recover source sessions for real long-context memory questions without using oracle answer labels.
2. Synthetic aftershock repair measures whether a learned dependency estimator can recover hidden downstream damage under controlled ground truth.

The project does not claim that the synthetic verifier is a real-world truth oracle. It is a controlled test harness for dependency repair algorithms.
