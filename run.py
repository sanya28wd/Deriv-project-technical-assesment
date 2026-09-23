from __future__ import annotations

from pathlib import Path

from rag_pipeline.pipeline import MODEL_NAME, PipelineError, run_pipeline


def main() -> None:
    root_directory: Path = Path(__file__).resolve().parent
    try:
        summary = run_pipeline(root_directory, MODEL_NAME)
    except PipelineError as error:
        raise SystemExit(f"Pipeline failed: {error}") from error
    print("Pipeline complete")
    print(f"Documents: {summary['documents']}")
    print(f"Chunks: {summary['chunks']}")
    print(f"Questions: {summary['questions']}")
    print(f"Supported answers: {summary['supported_answers']}")
    print(f"Validation failures: {summary['validation_failures']}")


if __name__ == "__main__":
    main()
