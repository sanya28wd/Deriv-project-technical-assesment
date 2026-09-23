from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag_pipeline.pipeline import (
    Answer,
    Chunk,
    PipelineError,
    PipelineStage,
    RetrievalResult,
    build_tfidf_index,
    create_chunks,
    combine_final_answers,
    create_unsupported_answer,
    load_documents,
    load_questions,
    retrieve,
    run_pipeline,
    UNSUPPORTED_ANSWER,
    validate_answer,
    advance_stage,
)


class PipelineTests(unittest.TestCase):
    def test_loads_multiple_supported_document_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root: Path = Path(temporary_directory)
            docs: Path = root / "docs"
            docs.mkdir()
            (docs / "first.md").write_text("Alpha", encoding="utf-8")
            (docs / "nested").mkdir()
            (docs / "nested" / "second.txt").write_text("Beta", encoding="utf-8")
            documents = load_documents(docs)
        self.assertEqual([document["path"] for document in documents], ["docs/first.md", "docs/nested/second.txt"])

    def test_replaced_knowledge_base_uses_runtime_paths_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root: Path = Path(temporary_directory)
            docs: Path = root / "docs"
            docs.mkdir()
            (docs / "renamed-policy.txt").write_text("Enterprise customers receive 500 requests per minute.", encoding="utf-8")
            (docs / "another-name.md").write_text("Webhook delivery history is retained for 14 days.", encoding="utf-8")
            documents = load_documents(docs)
            chunks = create_chunks(documents, 700, 120)
            index = build_tfidf_index(chunks)
            results = retrieve([{"id": "new-question", "question": "How many requests per minute do enterprise customers receive?"}], chunks, index, 3)
        self.assertEqual(documents[0]["doc_id"], "docs/another-name.md")
        self.assertEqual(results[0]["retrieved_chunks"][0]["doc_id"], "docs/renamed-policy.txt")

    def test_pipeline_stage_order_is_enforced(self) -> None:
        self.assertEqual(advance_stage(PipelineStage.INIT, PipelineStage.DOCUMENTS_LOADED), PipelineStage.DOCUMENTS_LOADED)
        with self.assertRaisesRegex(PipelineError, "Invalid stage transition"):
            advance_stage(PipelineStage.INIT, PipelineStage.ANSWERS_GENERATED)

    def test_chunks_have_stable_source_attribution_and_offsets(self) -> None:
        documents = [{"doc_id": "docs/a.txt", "path": "docs/a.txt", "text": "abcdefghij"}]
        chunks = create_chunks(documents, 6, 2)
        self.assertEqual(chunks[0], {"chunk_id": "docs/a.txt::chunk_1", "doc_id": "docs/a.txt", "text": "abcdef", "start_char": 0, "end_char": 6})
        self.assertEqual(chunks[1]["text"], "efghij")
        self.assertEqual(chunks[1]["start_char"], 4)

    def test_retrieval_returns_at_most_top_three_runtime_chunks(self) -> None:
        chunks: list[Chunk] = [
            {"chunk_id": "a", "doc_id": "a", "text": "OAuth authentication bearer token", "start_char": 0, "end_char": 31},
            {"chunk_id": "b", "doc_id": "b", "text": "webhook retention is thirty days", "start_char": 0, "end_char": 32},
            {"chunk_id": "c", "doc_id": "c", "text": "OAuth supports authorization", "start_char": 0, "end_char": 28},
            {"chunk_id": "d", "doc_id": "d", "text": "unrelated weather information", "start_char": 0, "end_char": 28},
        ]
        index = build_tfidf_index(chunks)
        results = retrieve([{"id": 1, "question": "Which OAuth authentication is supported?"}], chunks, index, 3)
        self.assertEqual(len(results[0]["retrieved_chunks"]), 3)
        self.assertEqual(results[0]["retrieved_chunks"][0]["chunk_id"], "a")

    def test_questions_reject_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            questions_path: Path = Path(temporary_directory) / "questions.json"
            questions_path.write_text(json.dumps([{"id": 1, "question": "one"}, {"id": 1, "question": "two"}]), encoding="utf-8")
            with self.assertRaisesRegex(PipelineError, "Duplicate question id"):
                load_questions(questions_path)

    def test_validation_rejects_unknown_citation(self) -> None:
        retrieval: RetrievalResult = {"id": 1, "question": "Question", "retrieved_chunks": [{"chunk_id": "known", "doc_id": "d", "score": 1.0, "text": "OAuth is supported"}]}
        answer: Answer = {"id": 1, "question": "Question", "answer": "OAuth is supported", "supported": True, "citations": ["missing"]}
        validation = validate_answer(answer, retrieval)
        self.assertFalse(validation["passed"])
        self.assertIn("Citations not retrieved", validation["issues"][0])

    def test_validation_rejects_unsupported_answer_with_citation(self) -> None:
        retrieval: RetrievalResult = {"id": 1, "question": "Question", "retrieved_chunks": [{"chunk_id": "known", "doc_id": "d", "score": 1.0, "text": "OAuth is supported"}]}
        answer: Answer = {"id": 1, "question": "Question", "answer": "Not available", "supported": False, "citations": ["known"]}
        validation = validate_answer(answer, retrieval)
        self.assertFalse(validation["passed"])
        self.assertIn("Unsupported answer must not include citations", validation["issues"])

    def test_validation_rejects_nonstandard_unsupported_answer(self) -> None:
        retrieval: RetrievalResult = {"id": 1, "question": "Question", "retrieved_chunks": [{"chunk_id": "known", "doc_id": "d", "score": 0.0, "text": "OAuth is supported"}]}
        answer: Answer = {"id": 1, "question": "Question", "answer": "I do not know.", "supported": False, "citations": []}
        validation = validate_answer(answer, retrieval)
        self.assertFalse(validation["passed"])
        self.assertTrue(any("Unsupported answer must use the required" in issue for issue in validation["issues"]))

    def test_irrelevant_question_has_zero_scores_and_required_refusal(self) -> None:
        chunks: list[Chunk] = [
            {"chunk_id": "auth", "doc_id": "auth", "text": "OAuth bearer tokens are supported", "start_char": 0, "end_char": 33},
            {"chunk_id": "logs", "doc_id": "logs", "text": "Webhook logs are retained for 30 days", "start_char": 0, "end_char": 37},
        ]
        index = build_tfidf_index(chunks)
        retrieval = retrieve([{"id": "unsupported", "question": "Which regions host the platform?"}], chunks, index, 3)[0]
        answer = create_unsupported_answer(retrieval)
        validation = validate_answer(answer, retrieval)
        self.assertTrue(all(chunk["score"] == 0.0 for chunk in retrieval["retrieved_chunks"]))
        self.assertEqual(answer["answer"], UNSUPPORTED_ANSWER)
        self.assertEqual(answer["citations"], [])
        self.assertTrue(validation["passed"])

    def test_all_unsupported_pipeline_run_needs_no_api_key_and_writes_empty_llm_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root: Path = Path(temporary_directory)
            docs: Path = root / "docs"
            docs.mkdir()
            (docs / "authentication.txt").write_text("OAuth bearer tokens are supported.", encoding="utf-8")
            (root / "questions.json").write_text(json.dumps([{"id": "unknown", "question": "Which galaxies host the platform?"}]), encoding="utf-8")
            summary = run_pipeline(root, "unused-model")
            answers = json.loads((root / "artifacts" / "answers.json").read_text(encoding="utf-8"))
            llm_log = (root / "artifacts" / "llm_calls.jsonl").read_text(encoding="utf-8")
        self.assertEqual(summary["supported_answers"], 0)
        self.assertEqual(answers[0]["answer"], UNSUPPORTED_ANSWER)
        self.assertEqual(llm_log, "")

    def test_validation_rejects_ungrounded_supported_answer(self) -> None:
        retrieval: RetrievalResult = {"id": 1, "question": "Question", "retrieved_chunks": [{"chunk_id": "known", "doc_id": "d", "score": 1.0, "text": "OAuth is supported"}]}
        answer: Answer = {"id": 1, "question": "Question", "answer": "Green elephants operate satellites", "supported": True, "citations": ["known"]}
        validation = validate_answer(answer, retrieval)
        self.assertFalse(validation["passed"])
        self.assertTrue(any("no meaningful term overlap" in issue for issue in validation["issues"]))

    def test_validation_rejects_supported_answer_without_citation(self) -> None:
        retrieval: RetrievalResult = {"id": 1, "question": "Question", "retrieved_chunks": [{"chunk_id": "known", "doc_id": "d", "score": 1.0, "text": "OAuth is supported"}]}
        answer: Answer = {"id": 1, "question": "Question", "answer": "OAuth is supported", "supported": True, "citations": []}
        validation = validate_answer(answer, retrieval)
        self.assertFalse(validation["passed"])
        self.assertIn("Supported answer must include at least one citation", validation["issues"])

    def test_validation_rejects_numeric_claim_not_in_cited_text(self) -> None:
        retrieval: RetrievalResult = {"id": 1, "question": "Question", "retrieved_chunks": [{"chunk_id": "known", "doc_id": "d", "score": 1.0, "text": "Logs are retained for 30 days"}]}
        answer: Answer = {"id": 1, "question": "Question", "answer": "Logs are retained for 60 days", "supported": True, "citations": ["known"]}
        validation = validate_answer(answer, retrieval)
        self.assertFalse(validation["passed"])
        self.assertIn("Numeric claims absent", validation["issues"][0])

    def test_final_answer_combines_retrieval_and_validation(self) -> None:
        retrieval: RetrievalResult = {"id": 1, "question": "Question", "retrieved_chunks": [{"chunk_id": "known", "doc_id": "d", "score": 1.0, "text": "OAuth is supported"}]}
        answer: Answer = {"id": 1, "question": "Question", "answer": "OAuth is supported", "supported": True, "citations": ["known"]}
        validation = validate_answer(answer, retrieval)
        final_answers = combine_final_answers([answer], [retrieval], [validation])
        self.assertEqual(final_answers[0]["retrieved_chunk_ids"], ["known"])
        self.assertEqual(final_answers[0]["validation"], validation)


if __name__ == "__main__":
    unittest.main()
