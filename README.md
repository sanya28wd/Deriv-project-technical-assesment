# Local Knowledge-Base Assistant

A small RAG pipeline that answers questions only from local `.md` and `.txt` documents and returns validated chunk citations.

## Run

1. Create and activate a virtual environment.
2. Install dependencies with `python3 -m pip install -r requirements.txt`.
3. Add `OPENAI_API_KEY=...` to `.env` (or export it in your shell).
4. Run `python3 run.py`.

The CLI reads every supported file under `docs/` and `questions.json`, then writes the following runtime artifacts:

- `artifacts/documents.json`
- `artifacts/chunks.json`
- `artifacts/retrieval_results.json`
- `artifacts/answers.json`
- `artifacts/citation_validation.json`
- `artifacts/final_answers.json`
- `artifacts/pipeline_stages.json`
- `artifacts/llm_calls.jsonl` (one audit record per OpenAI generation request)

### Replacing the knowledge base

Replace the contents of `docs/` with any `.md` and `.txt` files and replace `questions.json` with the same list-of-objects schema. Then run `python3 run.py` again. The pipeline discovers document paths at runtime, creates new stable chunk IDs from those paths, rebuilds the TF-IDF index, and does not depend on the bundled filenames, question order, or answers.

`artifacts/pipeline_stages.json` records the required stage order for each successful run. Restart `python3 web.py` after replacing documents so the browser UI rebuilds its in-memory index.

## Tests

Run `python3 -m unittest discover -s tests -v`.

The pipeline uses local TF-IDF/cosine similarity for retrieval and the OpenAI API only to turn the retrieved chunks into a constrained, cited answer. Do not commit `.env` or `artifacts/`.

Questions with no lexical overlap against any indexed chunk are deterministically returned as unsupported without an API call. All other answers are generated from retrieved chunks only and are checked for citation membership, supported/unsupported consistency, numeric grounding, and meaningful cited-term overlap.

## Browser UI

Run `python3 web.py`, then open <http://127.0.0.1:8000>. If that port is already in use, run `python3 web.py --port 8001` and open <http://127.0.0.1:8001>. The UI runs the same retrieval, grounded generation, and citation-validation flow for each question. It also displays index status, support/validation status, citation count, similarity metrics, and ranked source evidence. Use `Ctrl+C` in the terminal to stop it.

The latest fixture evaluation is summarized in [RESULTS.md](RESULTS.md).
