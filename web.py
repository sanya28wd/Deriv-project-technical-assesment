from __future__ import annotations

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Type

from rag_pipeline.pipeline import (
    MODEL_NAME,
    Answer,
    Chunk,
    PipelineError,
    RetrievalResult,
    TfidfIndex,
    build_tfidf_index,
    create_chunks,
    generate_answers,
    load_documents,
    load_environment,
    retrieve,
    validate_answer,
)


HOST: str = "127.0.0.1"
PORT: int = 8000
MAX_REQUEST_BYTES: int = 16_384


@dataclass(frozen=True)
class AssistantContext:
    """Read-only runtime data used by the HTTP interface."""

    document_count: int
    chunks: list[Chunk]
    index: TfidfIndex
    llm_log_path: Path


def load_assistant_context(root_directory: Path) -> AssistantContext:
    load_environment(root_directory / ".env")
    documents = load_documents(root_directory / "docs")
    chunks = create_chunks(documents, 700, 120)
    index = build_tfidf_index(chunks)
    return AssistantContext(
        document_count=len(documents),
        chunks=chunks,
        index=index,
        llm_log_path=root_directory / "artifacts" / "llm_calls.jsonl",
    )


def answer_metrics(answer: Answer, retrieval_result: RetrievalResult, validation_passed: bool) -> dict[str, int | float | bool]:
    scores: list[float] = [chunk["score"] for chunk in retrieval_result["retrieved_chunks"]]
    return {
        "retrieved_chunk_count": len(scores),
        "top_similarity_score": max(scores, default=0.0),
        "average_similarity_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "citation_count": len(answer["citations"]),
        "validation_passed": validation_passed,
    }


def answer_question(question: str, context: AssistantContext) -> dict[str, object]:
    normalized_question: str = question.strip()
    if not normalized_question:
        raise PipelineError("Question must not be empty")
    if len(normalized_question) > 2_000:
        raise PipelineError("Question must be 2,000 characters or fewer")
    retrieval_result: RetrievalResult = retrieve(
        [{"id": "interactive", "question": normalized_question}], context.chunks, context.index, 3
    )[0]
    answer: Answer = generate_answers([retrieval_result], MODEL_NAME, context.llm_log_path)[0]
    validation = validate_answer(answer, retrieval_result)
    return {
        "answer": answer["answer"],
        "supported": answer["supported"],
        "citations": answer["citations"],
        "retrieved_chunks": retrieval_result["retrieved_chunks"],
        "validation": validation,
        "metrics": answer_metrics(answer, retrieval_result, validation["passed"]),
    }


def load_index_html(root_directory: Path) -> bytes:
    html_path: Path = root_directory / "web" / "index.html"
    if not html_path.is_file():
        raise PipelineError(f"UI file does not exist: {html_path}")
    return html_path.read_bytes()


def create_request_handler(root_directory: Path, context: AssistantContext) -> Type[BaseHTTPRequestHandler]:
    index_html: bytes = load_index_html(root_directory)

    class RagRequestHandler(BaseHTTPRequestHandler):
        def send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            encoded_payload: bytes = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded_payload)))
            self.end_headers()
            self.wfile.write(encoded_payload)

        def do_GET(self) -> None:
            if self.path == "/api/status":
                self.send_json(HTTPStatus.OK, {
                    "document_count": context.document_count,
                    "chunk_count": len(context.chunks),
                    "retrieval_limit": 3,
                    "model": MODEL_NAME,
                })
                return
            if self.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(index_html)))
            self.end_headers()
            self.wfile.write(index_html)

        def do_POST(self) -> None:
            if self.path != "/api/ask":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            content_length: str | None = self.headers.get("Content-Length")
            if content_length is None or not content_length.isdigit():
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "A valid Content-Length header is required"})
                return
            request_size: int = int(content_length)
            if request_size > MAX_REQUEST_BYTES:
                self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Request body is too large"})
                return
            try:
                payload: object = json.loads(self.rfile.read(request_size).decode("utf-8"))
                if not isinstance(payload, dict) or not isinstance(payload.get("question"), str):
                    raise PipelineError("Request JSON must contain a string question")
                result: dict[str, object] = answer_question(payload["question"], context)
            except (UnicodeDecodeError, json.JSONDecodeError, PipelineError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except Exception as error:
                print(f"error=interactive_generation_failed detail={error}")
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Answer generation failed; check server logs."})
                return
            self.send_json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: object) -> None:
            print("http_request=" + (format % args))

    return RagRequestHandler


def parse_port() -> int:
    parser = ArgumentParser(description="Run the grounded knowledge-assistant UI")
    parser.add_argument("--port", type=int, default=PORT, help="Local TCP port for the UI")
    arguments = parser.parse_args()
    if arguments.port < 1 or arguments.port > 65_535:
        raise PipelineError("Port must be between 1 and 65535")
    return arguments.port


def main() -> None:
    root_directory: Path = Path(__file__).resolve().parent
    context: AssistantContext = load_assistant_context(root_directory)
    handler = create_request_handler(root_directory, context)
    port: int = parse_port()
    server = ThreadingHTTPServer((HOST, port), handler)
    print(f"RAG UI ready at http://{HOST}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("RAG UI stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
