from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

from openai import OpenAI


CHUNK_SIZE: int = 700
CHUNK_OVERLAP: int = 120
RETRIEVAL_LIMIT: int = 3
MINIMUM_RELEVANCE_SCORE: float = 0.0
MODEL_NAME: str = "gpt-4o-mini"
UNSUPPORTED_ANSWER: str = "The knowledge base does not provide enough information to answer this."
TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[a-zA-Z0-9]+")
NUMBER_PATTERN: re.Pattern[str] = re.compile(r"\b\d+(?:\.\d+)?\b")
STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "with", "you", "your",
})


class PipelineError(RuntimeError):
    """Raised when an input, model response, or pipeline invariant is invalid."""


class PipelineStage(StrEnum):
    """Required, observable stages of a complete pipeline run."""

    INIT = "INIT"
    DOCUMENTS_LOADED = "DOCUMENTS_LOADED"
    CHUNKS_CREATED = "CHUNKS_CREATED"
    INDEX_BUILT = "INDEX_BUILT"
    QUESTIONS_LOADED = "QUESTIONS_LOADED"
    RETRIEVAL_COMPLETE = "RETRIEVAL_COMPLETE"
    ANSWERS_GENERATED = "ANSWERS_GENERATED"
    CITATIONS_VALIDATED = "CITATIONS_VALIDATED"
    RESULTS_EXPORTED = "RESULTS_EXPORTED"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"


class Document(TypedDict):
    doc_id: str
    path: str
    text: str


class Chunk(TypedDict):
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int


class Question(TypedDict):
    id: int | str
    question: str


class RetrievedChunk(TypedDict):
    chunk_id: str
    doc_id: str
    score: float
    text: str


class RetrievalResult(TypedDict):
    id: int | str
    question: str
    retrieved_chunks: list[RetrievedChunk]


class Answer(TypedDict):
    id: int | str
    question: str
    answer: str
    supported: bool
    citations: list[str]


class ValidationResult(TypedDict):
    id: int | str
    passed: bool
    issues: list[str]


class FinalAnswer(Answer):
    retrieved_chunk_ids: list[str]
    validation: ValidationResult


class PipelineSummary(TypedDict):
    documents: int
    chunks: int
    questions: int
    supported_answers: int
    validation_failures: int


class TfidfIndex(TypedDict):
    document_frequency: Counter[str]
    chunk_vectors: dict[str, dict[str, float]]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def advance_stage(current_stage: PipelineStage, next_stage: PipelineStage) -> PipelineStage:
    stages: list[PipelineStage] = list(PipelineStage)
    current_index: int = stages.index(current_stage)
    expected_stage: PipelineStage | None = stages[current_index + 1] if current_index + 1 < len(stages) else None
    if next_stage != expected_stage:
        raise PipelineError(f"Invalid stage transition from {current_stage} to {next_stage}; expected {expected_stage}")
    return next_stage


def load_documents(docs_directory: Path) -> list[Document]:
    if not docs_directory.is_dir():
        raise PipelineError(f"Documents directory does not exist: {docs_directory}")

    source_paths: list[Path] = sorted(
        path for path in docs_directory.rglob("*") if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )
    if not source_paths:
        raise PipelineError(f"No .md or .txt files found under: {docs_directory}")

    documents: list[Document] = []
    for source_path in source_paths:
        relative_path: str = source_path.relative_to(docs_directory.parent).as_posix()
        document_text: str = source_path.read_text(encoding="utf-8").strip()
        if not document_text:
            raise PipelineError(f"Document is empty: {relative_path}")
        documents.append({"doc_id": relative_path, "path": relative_path, "text": document_text})
    return documents


def create_chunks(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise PipelineError("Chunk settings require chunk_size > chunk_overlap >= 0")

    chunks: list[Chunk] = []
    for document in documents:
        start_char: int = 0
        chunk_number: int = 1
        document_text: str = document["text"]
        while start_char < len(document_text):
            end_char: int = min(start_char + chunk_size, len(document_text))
            chunks.append({
                "chunk_id": f"{document['doc_id']}::chunk_{chunk_number}",
                "doc_id": document["doc_id"],
                "text": document_text[start_char:end_char],
                "start_char": start_char,
                "end_char": end_char,
            })
            if end_char == len(document_text):
                break
            start_char = end_char - chunk_overlap
            chunk_number += 1
    return chunks


def load_questions(questions_path: Path) -> list[Question]:
    if not questions_path.is_file():
        raise PipelineError(f"Questions file does not exist: {questions_path}")
    try:
        raw_questions: object = json.loads(questions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PipelineError(f"Questions file is not valid JSON: {error}") from error
    if not isinstance(raw_questions, list):
        raise PipelineError("questions.json must contain a JSON array")

    questions: list[Question] = []
    question_ids: set[int | str] = set()
    for index, raw_question in enumerate(raw_questions):
        if not isinstance(raw_question, dict):
            raise PipelineError(f"Question at index {index} must be an object")
        question_id: object = raw_question.get("id")
        question_text: object = raw_question.get("question")
        if not isinstance(question_id, (int, str)) or isinstance(question_id, bool):
            raise PipelineError(f"Question at index {index} has an invalid id")
        if not isinstance(question_text, str) or not question_text.strip():
            raise PipelineError(f"Question at index {index} must have a non-empty question")
        if question_id in question_ids:
            raise PipelineError(f"Duplicate question id: {question_id}")
        question_ids.add(question_id)
        questions.append({"id": question_id, "question": question_text.strip()})
    return questions


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def tfidf_vector(tokens: list[str], document_frequency: Counter[str], document_count: int) -> dict[str, float]:
    term_frequency: Counter[str] = Counter(tokens)
    if not term_frequency:
        return {}
    total_terms: int = len(tokens)
    return {
        term: (count / total_terms) * (math.log((document_count + 1) / (document_frequency[term] + 1)) + 1)
        for term, count in term_frequency.items()
    }


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    numerator: float = sum(weight * right.get(term, 0.0) for term, weight in left.items())
    left_norm: float = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm: float = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def build_tfidf_index(chunks: list[Chunk]) -> TfidfIndex:
    if not chunks:
        raise PipelineError("Cannot build an index without chunks")
    document_frequency: Counter[str] = Counter()
    chunk_tokens: dict[str, list[str]] = {}
    for chunk in chunks:
        tokens: list[str] = tokenize(chunk["text"])
        chunk_tokens[chunk["chunk_id"]] = tokens
        document_frequency.update(set(tokens))
    chunk_vectors: dict[str, dict[str, float]] = {
        chunk_id: tfidf_vector(tokens, document_frequency, len(chunks))
        for chunk_id, tokens in chunk_tokens.items()
    }
    return {"document_frequency": document_frequency, "chunk_vectors": chunk_vectors}


def retrieve(questions: list[Question], chunks: list[Chunk], index: TfidfIndex, limit: int) -> list[RetrievalResult]:
    if limit <= 0:
        raise PipelineError("Retrieval limit must be positive")

    results: list[RetrievalResult] = []
    for question in questions:
        question_vector: dict[str, float] = tfidf_vector(
            tokenize(question["question"]), index["document_frequency"], len(chunks)
        )
        scored_chunks: list[RetrievedChunk] = sorted(
            (
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "score": round(cosine_similarity(question_vector, index["chunk_vectors"][chunk["chunk_id"]]), 6),
                    "text": chunk["text"],
                }
                for chunk in chunks
            ),
            key=lambda item: (-item["score"], item["chunk_id"]),
        )[:limit]
        results.append({"id": question["id"], "question": question["question"], "retrieved_chunks": scored_chunks})
    return results


def load_environment(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped_line: str = line.strip()
        if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
            continue
        key, value = stripped_line.split("=", 1)
        normalized_key: str = key.strip()
        if normalized_key and normalized_key not in os.environ:
            os.environ[normalized_key] = value.strip().strip('"').strip("'")


def answer_schema() -> dict[str, object]:
    return {
        "type": "json_schema",
        "name": "grounded_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "supported": {"type": "boolean"},
                "citations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "supported", "citations"],
            "additionalProperties": False,
        },
    }


def build_prompt(retrieval_result: RetrievalResult) -> str:
    context: str = "\n\n".join(
        f"[{chunk['chunk_id']}]\n{chunk['text']}" for chunk in retrieval_result["retrieved_chunks"]
    )
    return f"""You are a grounded knowledge-base assistant. Return only JSON matching the supplied schema.
Answer only from the provided context. Do not use outside knowledge or infer product features, policies, or numbers.
If the context does not directly support an answer, set supported to false, use this exact answer: {UNSUPPORTED_ANSWER!r}, and return an empty citations array.
If supported is true, cite one or more returned chunk IDs and cite only a chunk that directly supports the answer.

Question: {retrieval_result['question']}

Context:
{context}
"""


def parse_model_answer(raw_output: str, retrieval_result: RetrievalResult) -> Answer:
    try:
        parsed_output: object = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise PipelineError(f"Model returned invalid JSON for question {retrieval_result['id']}: {raw_output!r}") from error
    if not isinstance(parsed_output, dict):
        raise PipelineError(f"Model response for question {retrieval_result['id']} must be an object: {raw_output!r}")
    answer_text: object = parsed_output.get("answer")
    supported: object = parsed_output.get("supported")
    citations: object = parsed_output.get("citations")
    if not isinstance(answer_text, str) or not answer_text.strip():
        raise PipelineError(f"Model answer for question {retrieval_result['id']} is missing a non-empty answer")
    if not isinstance(supported, bool):
        raise PipelineError(f"Model answer for question {retrieval_result['id']} has a non-boolean supported value")
    if not isinstance(citations, list) or not all(isinstance(citation, str) for citation in citations):
        raise PipelineError(f"Model answer for question {retrieval_result['id']} has invalid citations")
    return {
        "id": retrieval_result["id"],
        "question": retrieval_result["question"],
        "answer": answer_text.strip(),
        "supported": supported,
        "citations": citations,
    }


def create_unsupported_answer(retrieval_result: RetrievalResult) -> Answer:
    return {
        "id": retrieval_result["id"],
        "question": retrieval_result["question"],
        "answer": UNSUPPORTED_ANSWER,
        "supported": False,
        "citations": [],
    }


def has_relevant_retrieval(retrieval_result: RetrievalResult, minimum_score: float) -> bool:
    return any(chunk["score"] > minimum_score for chunk in retrieval_result["retrieved_chunks"])


def write_llm_call_log(log_path: Path, retrieval_result: RetrievalResult, model_name: str, prompt: str) -> None:
    append_jsonl(log_path, {
        "stage": "answer_generation",
        "timestamp": datetime.now(UTC).isoformat(),
        "provider": "openai",
        "model": model_name,
        "question_id": retrieval_result["id"],
        "prompt_hash": sha256(prompt.encode("utf-8")).hexdigest(),
        "input_artifacts": ["artifacts/retrieval_results.json"],
        "output_artifact": "artifacts/answers.json",
    })


def generate_answers(retrieval_results: list[RetrievalResult], model_name: str, llm_log_path: Path) -> list[Answer]:
    unsupported_answers: dict[int | str, Answer] = {
        result["id"]: create_unsupported_answer(result)
        for result in retrieval_results
        if not has_relevant_retrieval(result, MINIMUM_RELEVANCE_SCORE)
    }
    relevant_results: list[RetrievalResult] = [
        result for result in retrieval_results if result["id"] not in unsupported_answers
    ]
    if not relevant_results:
        return [unsupported_answers[result["id"]] for result in retrieval_results]
    api_key: str | None = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise PipelineError("OPENAI_API_KEY is required. Add it to .env or export it in your shell.")
    client = OpenAI(api_key=api_key)
    answers: list[Answer] = []
    generated_answers: dict[int | str, Answer] = {}
    for retrieval_result in relevant_results:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                prompt: str = build_prompt(retrieval_result)
                write_llm_call_log(llm_log_path, retrieval_result, model_name, prompt)
                response = client.responses.create(
                    model=model_name,
                    input=prompt,
                    text={"format": answer_schema()},
                )
                if not response.output_text:
                    raise PipelineError(f"Model returned empty output for question {retrieval_result['id']}")
                generated_answers[retrieval_result["id"]] = parse_model_answer(response.output_text, retrieval_result)
                last_error = None
                break
            except Exception as error:
                last_error = error
                if attempt < 3:
                    print(f"Warning: generation attempt {attempt} failed for question {retrieval_result['id']}; retrying.")
                    time.sleep(attempt)
        if last_error is not None:
            raise PipelineError(
                f"Model generation failed for question {retrieval_result['id']} after 3 attempts: {last_error}"
            ) from last_error
    answers.extend(unsupported_answers.get(result["id"], generated_answers[result["id"]]) for result in retrieval_results)
    return answers


def normalize_grounding_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) > 2 and token not in STOPWORDS}


def validate_answer(answer: Answer, retrieval_result: RetrievalResult) -> ValidationResult:
    issues: list[str] = []
    retrieved_by_id: dict[str, RetrievedChunk] = {
        chunk["chunk_id"]: chunk for chunk in retrieval_result["retrieved_chunks"]
    }
    unknown_citations: list[str] = [citation for citation in answer["citations"] if citation not in retrieved_by_id]
    if unknown_citations:
        issues.append(f"Citations not retrieved for this question: {unknown_citations}")
    if answer["supported"] and not answer["citations"]:
        issues.append("Supported answer must include at least one citation")
    if not answer["supported"] and answer["citations"]:
        issues.append("Unsupported answer must not include citations")
    if not answer["supported"] and answer["answer"] != UNSUPPORTED_ANSWER:
        issues.append("Unsupported answer must use the required unavailable-information message")
    if answer["supported"] and not unknown_citations:
        cited_text: str = " ".join(retrieved_by_id[citation]["text"] for citation in answer["citations"])
        cited_numbers: set[str] = set(NUMBER_PATTERN.findall(cited_text))
        answer_numbers: set[str] = set(NUMBER_PATTERN.findall(answer["answer"]))
        missing_numbers: set[str] = answer_numbers - cited_numbers
        if missing_numbers:
            issues.append(f"Numeric claims absent from cited text: {sorted(missing_numbers)}")
        answer_terms: set[str] = normalize_grounding_tokens(answer["answer"])
        cited_terms: set[str] = normalize_grounding_tokens(cited_text)
        if answer_terms and not answer_terms.intersection(cited_terms):
            issues.append("Answer has no meaningful term overlap with cited text")
    return {"id": answer["id"], "passed": not issues, "issues": issues}


def validate_answers(answers: list[Answer], retrieval_results: list[RetrievalResult]) -> list[ValidationResult]:
    retrieval_by_id: dict[int | str, RetrievalResult] = {result["id"]: result for result in retrieval_results}
    validations: list[ValidationResult] = []
    for answer in answers:
        retrieval_result: RetrievalResult | None = retrieval_by_id.get(answer["id"])
        if retrieval_result is None:
            raise PipelineError(f"No retrieval result found for answer {answer['id']}")
        validations.append(validate_answer(answer, retrieval_result))
    return validations


def combine_final_answers(
    answers: list[Answer], retrieval_results: list[RetrievalResult], validations: list[ValidationResult]
) -> list[FinalAnswer]:
    retrieval_by_id: dict[int | str, RetrievalResult] = {result["id"]: result for result in retrieval_results}
    validation_by_id: dict[int | str, ValidationResult] = {result["id"]: result for result in validations}
    return [
        {
            **answer,
            "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in retrieval_by_id[answer["id"]]["retrieved_chunks"]],
            "validation": validation_by_id[answer["id"]],
        }
        for answer in answers
    ]


def run_pipeline(root_directory: Path, model_name: str) -> PipelineSummary:
    artifacts_directory: Path = root_directory / "artifacts"
    current_stage: PipelineStage = PipelineStage.INIT
    completed_stages: list[str] = [current_stage.value]
    documents: list[Document] = load_documents(root_directory / "docs")
    current_stage = advance_stage(current_stage, PipelineStage.DOCUMENTS_LOADED)
    completed_stages.append(current_stage.value)
    write_json(artifacts_directory / "documents.json", documents)
    chunks: list[Chunk] = create_chunks(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    current_stage = advance_stage(current_stage, PipelineStage.CHUNKS_CREATED)
    completed_stages.append(current_stage.value)
    write_json(artifacts_directory / "chunks.json", chunks)
    index: TfidfIndex = build_tfidf_index(chunks)
    current_stage = advance_stage(current_stage, PipelineStage.INDEX_BUILT)
    completed_stages.append(current_stage.value)
    questions: list[Question] = load_questions(root_directory / "questions.json")
    current_stage = advance_stage(current_stage, PipelineStage.QUESTIONS_LOADED)
    completed_stages.append(current_stage.value)
    retrieval_results: list[RetrievalResult] = retrieve(questions, chunks, index, RETRIEVAL_LIMIT)
    current_stage = advance_stage(current_stage, PipelineStage.RETRIEVAL_COMPLETE)
    completed_stages.append(current_stage.value)
    write_json(artifacts_directory / "retrieval_results.json", retrieval_results)
    load_environment(root_directory / ".env")
    llm_log_path: Path = artifacts_directory / "llm_calls.jsonl"
    llm_log_path.parent.mkdir(parents=True, exist_ok=True)
    llm_log_path.write_text("", encoding="utf-8")
    answers: list[Answer] = generate_answers(retrieval_results, model_name, llm_log_path)
    current_stage = advance_stage(current_stage, PipelineStage.ANSWERS_GENERATED)
    completed_stages.append(current_stage.value)
    write_json(artifacts_directory / "answers.json", answers)
    validations: list[ValidationResult] = validate_answers(answers, retrieval_results)
    current_stage = advance_stage(current_stage, PipelineStage.CITATIONS_VALIDATED)
    completed_stages.append(current_stage.value)
    write_json(artifacts_directory / "citation_validation.json", validations)
    final_answers: list[FinalAnswer] = combine_final_answers(answers, retrieval_results, validations)
    write_json(artifacts_directory / "final_answers.json", final_answers)
    current_stage = advance_stage(current_stage, PipelineStage.RESULTS_EXPORTED)
    completed_stages.append(current_stage.value)
    current_stage = advance_stage(current_stage, PipelineStage.VALIDATION_COMPLETE)
    completed_stages.append(current_stage.value)
    write_json(artifacts_directory / "pipeline_stages.json", {"completed_stages": completed_stages})
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "questions": len(questions),
        "supported_answers": sum(1 for answer in answers if answer["supported"]),
        "validation_failures": sum(1 for validation in validations if not validation["passed"]),
    }
