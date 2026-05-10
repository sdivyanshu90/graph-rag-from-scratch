from __future__ import annotations

import json

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
                answer="alice worked with acme corp on flood sensors in river county.",
                context_text="[doc-1:0 | doc-1] Alice worked with Acme Corp on flood sensors in River County.",
                retrieved_chunks=[
                    RetrievedChunk(
                        chunk_id="doc-1:0",
                        source_id="doc-1",
                        text="Alice worked with Acme Corp on flood sensors in River County.",
                    )
                ],
                provenance=QueryProvenance(
                    node_names=["Alice", "Acme Corp"],
                    chunk_ids=["doc-1:0"],
                    source_ids=["doc-1"],
                ),
            )

        return QueryResult(
            mode="global",
            question=question,
            answer="The main themes are flood resilience work and grid operations.",
            context_text="Community 0 summary: flood resilience. Community 1 summary: grid operations.",
            provenance=QueryProvenance(
                node_names=["Alice", "Solar Lab"],
                community_ids=[0, 1],
                source_ids=["doc-1", "doc-2"],
            ),
        )


class FakeJudgeModel:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        if "grid operations" in user_prompt:
            return json.dumps(
                {
                    "faithfulness": 4,
                    "relevance": 5,
                    "justification": "The answer reflects the retrieved summaries and answers the question directly.",
                }
            )

        return json.dumps(
            {
                "faithfulness": 5,
                "relevance": 4,
                "justification": "The answer is fully supported by the retrieved chunk and close to the expected answer.",
            }
        )


def test_evaluation_harness_computes_exact_match_and_judge_summary() -> None:
    harness = EvaluationHarness(
        query_service=FakeQueryService(),
        answer_judge=LLMAnswerJudge(FakeJudgeModel()),
    )
    cases = [
        EvaluationCase(
            case_id="case-1",
            question="What did Alice do in River County?",
            expected_answer="Alice worked with Acme Corp on flood sensors in River County.",
            mode="local",
        ),
        EvaluationCase(
            case_id="case-2",
            question="What are the main themes?",
            expected_answer="The main themes are flood resilience work and grid operations.",
            mode="global",
        ),
    ]

    report = harness.run_cases(fixture_name="test-fixture", cases=cases)

    assert report.fixture_name == "test-fixture"
    assert report.summary.total_cases == 2
    assert report.summary.successful_cases == 2
    assert report.summary.exact_match_count == 2
    assert report.summary.exact_match_rate == 1.0
    assert report.summary.average_faithfulness == 4.5
    assert report.summary.average_relevance == 4.5
    assert report.results[0].judge_score is not None
    assert report.results[0].provenance is not None


def test_evaluation_harness_records_case_errors_without_stopping() -> None:
    class FailingQueryService:
        def query(
            self,
            *,
            question: str,
            mode: str,
            top_k: int | None = None,
            max_hops: int | None = None,
            max_chunks: int | None = None,
        ) -> QueryResult:
            del question
            del mode
            del top_k
            del max_hops
            del max_chunks
            raise ValueError("graph is not ready")

    harness = EvaluationHarness(query_service=FailingQueryService(), answer_judge=None)
    report = harness.run_cases(
        fixture_name="failing-fixture",
        cases=[
            EvaluationCase(
                case_id="case-1",
                question="What happened?",
                expected_answer="Something happened.",
                mode="local",
            )
        ],
    )

    assert report.summary.total_cases == 1
    assert report.summary.successful_cases == 0
    assert report.summary.exact_match_rate == 0.0
    assert report.results[0].error == "graph is not ready"