from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_rag.evaluation import EvaluationCase, EvaluationHarness, LLMAnswerJudge
from graph_rag.models import QueryProvenance, QueryResult, RetrievedChunk


class FakeQueryService:
    def query(
        self,
        *,
        question: str,
        mode: str,
        top_k: int | None = None,
        max_hops: int | None = None,
        max_chunks: int | None = None,
    ) -> QueryResult:
        del top_k
        del max_hops
        del max_chunks

        if mode == "local":
            return QueryResult(
                mode="local",
                question=question,
                answer="Alice worked with Acme Corp on flood sensors in River County.",
                context_text="Supporting chunk excerpts:\n[doc-1:0 | doc-1] Alice worked with Acme Corp on flood sensors in River County.",
                retrieved_chunks=[
                    RetrievedChunk(
                        chunk_id="doc-1:0",
                        source_id="doc-1",
                        text="Alice worked with Acme Corp on flood sensors in River County.",
                    )
                ],
                provenance=QueryProvenance(
                    node_names=["Alice", "Acme Corp", "River County"],
                    chunk_ids=["doc-1:0"],
                    source_ids=["doc-1"],
                ),
            )

        return QueryResult(
            mode="global",
            question=question,
            answer="The main themes are flood resilience work and grid operations.",
            context_text="Selected community reports:\nCommunity 0 ... flood resilience\nCommunity 1 ... grid operations",
            provenance=QueryProvenance(
                node_names=["Alice", "Acme Corp", "River County", "Solar Lab"],
                community_ids=[0, 1],
                source_ids=["doc-1", "doc-2"],
            ),
        )


class FakeJudgeModel:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        if "flood resilience" in user_prompt:
            return json.dumps(
                {
                    "faithfulness": 5,
                    "relevance": 5,
                    "justification": "The answer matches the provided community summaries and addresses the question directly.",
                }
            )

        return json.dumps(
            {
                "faithfulness": 5,
                "relevance": 4,
                "justification": "The answer is supported by the chunk excerpt and mostly matches the expected wording.",
            }
        )


def main() -> None:
    harness = EvaluationHarness(
        query_service=FakeQueryService(),
        answer_judge=LLMAnswerJudge(FakeJudgeModel()),
    )
    cases = [
        EvaluationCase(
            case_id="local-1",
            question="What did Alice do in River County?",
            expected_answer="Alice worked with Acme Corp on flood sensors in River County.",
            mode="local",
            top_k=2,
            max_hops=1,
            max_chunks=3,
        ),
        EvaluationCase(
            case_id="global-1",
            question="What are the main themes?",
            expected_answer="The main themes are flood resilience work and grid operations.",
            mode="global",
            top_k=2,
        ),
    ]
    report = harness.run_cases(fixture_name="phase7-demo", cases=cases)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()