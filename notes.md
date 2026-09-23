# Project Notes: Grounded Local RAG Assistant

This project builds a small Retrieval-Augmented Generation (RAG) assistant for a local developer-platform knowledge base. Its purpose is to answer a question only when the answer is supported by the supplied documents. When the evidence is insufficient, it returns the required unavailable-information response rather than inventing an answer.

## Pipeline

1. Load every `.md` and `.txt` file from `docs/` at runtime.
2. Split each document into source-attributed chunks with stable IDs and character offsets.
3. Build a local TF-IDF index and use cosine similarity to retrieve the three most relevant chunks for each question.
4. Send only the retrieved chunks to the LLM with strict instructions to answer from context, return structured JSON, and cite chunk IDs.
5. For questions with no lexical retrieval evidence, return the deterministic unsupported response without calling the LLM.
6. Validate every answer after generation: citations must come from retrieved chunks; supported answers need citations; unsupported answers must not have citations; numeric claims and meaningful terms are checked against the cited content.

The project deliberately uses TF-IDF rather than embeddings. This keeps retrieval local, transparent, inexpensive, and reproducible when the evaluator replaces the knowledge-base files.

## Evidence and observability

The pipeline writes machine-readable intermediate results to `artifacts/`, including documents, chunks, retrieval results, answers, citation validation, final answers, pipeline-stage trace, and LLM call audit logs. The stages are explicitly recorded from `INIT` through `VALIDATION_COMPLETE`.

## How to run

```bash
python3 -m pip install -r requirements.txt
python3 run.py
python3 -m unittest discover -s tests -v
python3 web.py
```

Open `http://127.0.0.1:8000` after starting the UI. To test a replacement knowledge base, replace the files inside `docs/` and replace `questions.json` using the same `{ "id", "question" }` schema, then rerun the pipeline or restart the UI.

## Current verification

- 15 automated tests pass.
- The fixture end-to-end run produced 0 citation-validation failures.
- The project includes supported, unsupported, invalid-citation, missing-citation, numeric-grounding, and replacement-knowledge-base checks.
