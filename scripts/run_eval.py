from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_rag.api_service import GraphRAGAPIService
from graph_rag.config import get_settings
from graph_rag.evaluation import EvaluationFixture, EvaluationHarness, LLMAnswerJudge
from graph_rag.llm import OpenAIChatClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Graph RAG evaluation cases and write a JSON report.")
    parser.add_argument("--fixture", required=True, help="Path to an evaluation fixture JSON file.")
    parser.add_argument("--output", required=True, help="Path to write the JSON report.")
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the LLM-as-judge step and compute exact match only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixture_path = Path(args.fixture)
    output_path = Path(args.output)
    fixture = EvaluationFixture.model_validate(json.loads(fixture_path.read_text(encoding="utf-8")))
    settings = get_settings()

    query_service = GraphRAGAPIService.from_settings(settings)
    try:
        for document in fixture.documents:
            query_service.ingest_text(source_id=document.source_id, text=document.text)

        answer_judge = None
        if not args.skip_judge:
            if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value().strip():
                raise ValueError("OPENAI_API_KEY is required unless --skip-judge is used")

            judge_client = OpenAIChatClient(
                model=settings.eval_judge_model,
                api_key=settings.openai_api_key.get_secret_value(),
                base_url=settings.openai_base_url,
            )
            answer_judge = LLMAnswerJudge(judge_client)

        harness = EvaluationHarness(query_service=query_service, answer_judge=answer_judge)
        report = harness.run_cases(fixture_name=fixture.name, cases=fixture.cases)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

        print(f"Wrote evaluation report to {output_path}")
        print(
            json.dumps(
                {
                    "fixture_name": report.fixture_name,
                    "total_cases": report.summary.total_cases,
                    "successful_cases": report.summary.successful_cases,
                    "exact_match_rate": report.summary.exact_match_rate,
                    "average_faithfulness": report.summary.average_faithfulness,
                    "average_relevance": report.summary.average_relevance,
                },
                indent=2,
            )
        )
        return 0
    finally:
        query_service.close()


if __name__ == "__main__":
    raise SystemExit(main())