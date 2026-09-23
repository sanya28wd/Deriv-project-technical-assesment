# Evaluation Results

## Run summary

The end-to-end pipeline was run against the bundled fictional knowledge base on 23 September 2026.

| Metric | Result |
| --- | ---: |
| Source documents loaded | 3 |
| Chunks created | 3 |
| Questions processed | 7 |
| Supported answers | 6 |
| Unsupported answers | 1 |
| Citation-validation failures | 0 |
| Automated tests passing | 15 / 15 |

## Answer evaluation

| ID | Question | Result | Citation |
| ---: | --- | --- | --- |
| 1 | What authentication methods does the API support? | Personal access tokens and OAuth 2.0 bearer tokens. | `docs/api.md::chunk_1` |
| 2 | How long are webhook delivery logs retained? | 30 days. | `docs/webhooks.txt::chunk_1` |
| 3 | Is there a free plan for startups? | Yes—there is a free Startup plan for companies with fewer than ten employees. | `docs/plans_and_data.md::chunk_1` |
| 4 | What happens if I exceed the rate limit? | Requests receive HTTP 429; clients should wait and retry using exponential backoff. | `docs/api.md::chunk_1` |
| 5 | Can I delete customer data immediately on request? | No—deletion is processed within 30 days after identity verification. | `docs/plans_and_data.md::chunk_1` |
| 6 | Does the platform support GraphQL subscriptions? | No. | `docs/api.md::chunk_1` |
| 7 | Which regions host the platform? | Not supported by the knowledge base. | None |

## Validation coverage

The test suite verifies document ingestion, stable chunk IDs and offsets, top-three retrieval, malformed question rejection, citation membership, supported and unsupported citation rules, numeric-claim grounding, and final-output composition.

Generated machine-readable evidence is available under `artifacts/` after `python3 run.py`.

The pipeline also exports `artifacts/pipeline_stages.json`, which records the required stage sequence from `INIT` through `VALIDATION_COMPLETE`.

LLM requests are recorded without prompt content or secrets in `artifacts/llm_calls.jsonl`; each record includes the model, question ID, UTC timestamp, prompt hash, and the relevant input/output artifact names.
