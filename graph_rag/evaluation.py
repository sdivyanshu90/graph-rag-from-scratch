from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from .models import QueryProvenance, QueryResult

JUDGE_SYSTEM_PROMPT = """
You are grading a Graph RAG answer.

Return JSON only with this exact shape:
{
  "faithfulness": 1,
  "relevance": 1,
  "justification": "short explanation"
}

Scoring rubric:
- faithfulness: 1 means unsupported by retrieved context, 3 means partially supported or mixed, 5 means fully supported by retrieved context.
- relevance: 1 means it misses the question badly, 3 means partially answers it, 5 means directly and usefully answers it.
- Use integers only for both scores.
- Keep justification brief and concrete.
""".strip()

WHITESPACE_PATTERN = re.compile(r"\s+")


class QueryService(Protocol):
    def query(
        self,
        *,
        question: str,
        mode: str,
        top_k: int | None = None,
        max_hops: int | None = None,
        max_chunks: int | None = None,
    ) -> QueryResult:
        ...


class JsonJudgeModel(Protocol):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


class EvaluationDocument(BaseModel):
    source_id: str
    text: str

    @field_validator("source_id", "text")
    @classmethod
    def validate_required_fields(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("evaluation document fields must not be empty")
        return cleaned_value


class EvaluationCase(BaseModel):
    case_id: str
    question: str
    expected_answer: str
    mode: Literal["local", "global"] = "local"
    top_k: int | None = None
    max_hops: int | None = None
    max_chunks: int | None = None

    @field_validator("case_id", "question", "expected_answer")
    @classmethod
    def validate_case_fields(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("evaluation case fields must not be empty")
        return cleaned_value


class EvaluationFixture(BaseModel):
    name: str = "graph-rag-eval"
    documents: list[EvaluationDocument] = Field(default_factory=list)
    cases: list[EvaluationCase] = Field(default_factory=list)


class JudgeScore(BaseModel):
    faithfulness: int
    relevance: int
    justification: str

    @field_validator("faithfulness", "relevance")
    @classmethod
    def validate_score_range(cls, value: int) -> int:
        if value < 1 or value > 5:
            raise ValueError("judge scores must be between 1 and 5")
        return value

    @field_validator("justification")
    @classmethod
    def validate_justification(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("justification must not be empty")
        return cleaned_value


class EvaluationCaseResult(BaseModel):
    case_id: str
    question: str
    mode: Literal["local", "global"]
    expected_answer: str
    actual_answer: str | None = None
    exact_match: bool = False
    judge_score: JudgeScore | None = None
    provenance: QueryProvenance | None = None
    error: str | None = None


class EvaluationSummary(BaseModel):
    total_cases: int
    successful_cases: int
    exact_match_count: int
    exact_match_rate: float
    average_faithfulness: float | None = None
    average_relevance: float | None = None


class EvaluationReport(BaseModel):
    generated_at: str
    fixture_name: str
    results: list[EvaluationCaseResult] = Field(default_factory=list)
    summary: EvaluationSummary


class LLMAnswerJudge:
    def __init__(self, judge_model: JsonJudgeModel) -> None:
        self.judge_model = judge_model

    def judge(
        self,
        *,
        question: str,
        expected_answer: str,
        actual_answer: str,
        context_text: str,
    ) -> JudgeScore:
        raw_response = self.judge_model.complete_json(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=(
                f"Question:\n{question}\n\n"
                f"Expected answer:\n{expected_answer}\n\n"
                f"Actual answer:\n{actual_answer}\n\n"
                f"Retrieved context:\n{context_text}"
            ),
        )
        return JudgeScore.model_validate_json(raw_response)


class EvaluationHarness:
    def __init__(
        self,
        *,
        query_service: QueryService,
        answer_judge: LLMAnswerJudge | None = None,
    ) -> None:
        self.query_service = query_service
        self.answer_judge = answer_judge

    def run_cases(
        self,
        *,
        fixture_name: str,
        cases: list[EvaluationCase],
    ) -> EvaluationReport:
        results: list[EvaluationCaseResult] = []

        for case in cases:
            try:
                query_result = self.query_service.query(
                    question=case.question,
                    mode=case.mode,
                    top_k=case.top_k,
                    max_hops=case.max_hops,
                    max_chunks=case.max_chunks,
                )
                exact_match = self._normalized_exact_match(
                    case.expected_answer,
                    query_result.answer,
                )
                judge_score = None
                if self.answer_judge is not None:
                    judge_score = self.answer_judge.judge(
                        question=case.question,
                        expected_answer=case.expected_answer,
                        actual_answer=query_result.answer,
                        context_text=query_result.context_text,
                    )

                results.append(
                    EvaluationCaseResult(
                        case_id=case.case_id,
                        question=case.question,
                        mode=case.mode,
                        expected_answer=case.expected_answer,
                        actual_answer=query_result.answer,
                        exact_match=exact_match,
                        judge_score=judge_score,
                        provenance=query_result.provenance,
                    )
                )
            except Exception as error:
                results.append(
                    EvaluationCaseResult(
                        case_id=case.case_id,
                        question=case.question,
                        mode=case.mode,
                        expected_answer=case.expected_answer,
                        error=str(error),
                    )
                )

        return EvaluationReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            fixture_name=fixture_name,
            results=results,
            summary=self._build_summary(results),
        )

    @staticmethod
    def _build_summary(results: list[EvaluationCaseResult]) -> EvaluationSummary:
        total_cases = len(results)
        successful_results = [result for result in results if result.error is None]
        exact_match_count = sum(1 for result in successful_results if result.exact_match)
        judged_results = [result for result in successful_results if result.judge_score is not None]

        average_faithfulness = None
        average_relevance = None
        if judged_results:
            average_faithfulness = sum(
                result.judge_score.faithfulness
                for result in judged_results
                if result.judge_score is not None
            ) / len(judged_results)
            average_relevance = sum(
                result.judge_score.relevance
                for result in judged_results
                if result.judge_score is not None
            ) / len(judged_results)

        return EvaluationSummary(
            total_cases=total_cases,
            successful_cases=len(successful_results),
            exact_match_count=exact_match_count,
            exact_match_rate=(exact_match_count / len(successful_results)) if successful_results else 0.0,
            average_faithfulness=average_faithfulness,
            average_relevance=average_relevance,
        )

    @staticmethod
    def _normalized_exact_match(expected_answer: str, actual_answer: str) -> bool:
        return EvaluationHarness._normalize_text(expected_answer) == EvaluationHarness._normalize_text(actual_answer)

    @staticmethod
    def _normalize_text(value: str) -> str:
        return WHITESPACE_PATTERN.sub(" ", value.strip()).casefold()