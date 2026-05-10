# Phase 7 - Evaluation Harness

Up to this point, we built a Graph RAG system that can ingest text, enrich a graph, answer local questions, answer global questions, sync to Neo4j, and expose an API.

Now we need one more professional habit: measuring whether the system is actually doing a good job.

Why this phase exists:

- demos can look impressive while still hiding brittle behavior
- retrieval systems often fail quietly by returning plausible but weak answers
- once you start changing prompts, chunk sizes, top-k values, or graph logic, you need a repeatable way to compare runs

Analogy first:

Think of evaluation like a quiz with an answer key and a grader.

- the answer key gives you one blunt check: did the system say the expected thing?
- the grader gives you a softer check: even if the wording changed, was the answer still grounded and useful?

That is exactly why this phase uses two signals:

- exact match
- LLM-as-judge scores for faithfulness and relevance

## 1. Keep the eval path honest

Before the code, the most important design decision:

We do not build a second query path just for evaluation.

Instead, the evaluation harness calls the same query interface the rest of the project already uses.

```python
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
```

What just happened?

We defined evaluation as a wrapper around the real system, not as a separate experiment-only implementation. That keeps the scores honest.

Design note:

This matters more than it may seem. If your evaluation code bypasses the real retrieval path, your numbers stop describing production behavior.

## 2. Model each evaluation case explicitly

Each eval case is just a small contract.

```python
class EvaluationCase(BaseModel):
    case_id: str
    question: str
    expected_answer: str
    mode: Literal["local", "global"] = "local"
    top_k: int | None = None
    max_hops: int | None = None
    max_chunks: int | None = None
```

What just happened?

We turned an evaluation row into typed data. That means each case can control not only the question, but also the retrieval settings used for that question.

Why include retrieval settings in the case?

Because evaluation is often where you compare system choices. For example:

- local search with `top_k=2` versus `top_k=5`
- one-hop neighborhoods versus two-hop neighborhoods
- local search versus global search for the same corpus

Common beginner mistake:

- storing only question and expected answer, then forgetting which retrieval settings produced the score

## 3. Start with exact match

Exact match is intentionally simple.

```python
def _normalize_text(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value.strip()).casefold()


def _normalized_exact_match(expected_answer: str, actual_answer: str) -> bool:
    return _normalize_text(expected_answer) == _normalize_text(actual_answer)
```

What just happened?

We normalize whitespace and letter case, then compare the two strings directly.

Why bother with exact match if it is so blunt?

- it is cheap
- it is deterministic
- it catches obvious regressions fast
- it gives you a stable baseline before you add softer metrics

Design note:

Exact match is useful precisely because it is strict. If a prompt change suddenly drops exact-match rate, that is a strong signal something changed. But it should not be your only metric because a correct paraphrase can still fail exact match.

## 4. Add an LLM judge for the soft part

Now the more interesting metric.

We ask another LLM to score the answer on two dimensions:

- faithfulness: is the answer supported by the retrieved context?
- relevance: does the answer actually address the question well?

Here is the judge shape:

```python
class JudgeScore(BaseModel):
    faithfulness: int
    relevance: int
    justification: str


class LLMAnswerJudge:
    def judge(
        self,
        *,
        question: str,
        expected_answer: str,
        actual_answer: str,
        context_text: str,
    ) -> JudgeScore:
        ...
```

What just happened?

We separated answer generation from answer grading. The query model produces the answer. The judge model critiques the result using the question, expected answer, and retrieved evidence.

Why has LLM-as-judge become common?

- exact match misses valid paraphrases
- many RAG answers are open-ended summaries, not short fact strings
- human review does not scale well once you have dozens or hundreds of cases
- LLM judges can evaluate grounding and usefulness in ways string metrics cannot

But be careful. LLM-as-judge has real weaknesses:

- prompt sensitivity: small wording changes in the judge prompt can change scores
- model bias: some judge models are too generous, too harsh, or style-biased
- non-determinism: repeated runs may drift unless temperature is pinned and prompting is stable
- cost and latency: scoring with another LLM adds expense and runtime
- false confidence: a judge can sound thoughtful while still being wrong

Design note:

Treat judge scores as one instrument on your dashboard, not as absolute truth. In a serious system, you would combine them with human spot checks and task-specific metrics.

## 5. Run the harness and emit a report

The harness loops over cases, calls the existing query path, computes exact match, optionally asks the judge, and returns a structured report.

```python
class EvaluationHarness:
    def run_cases(
        self,
        *,
        fixture_name: str,
        cases: list[EvaluationCase],
    ) -> EvaluationReport:
        ...
```

The report includes:

- per-case answers
- exact match flags
- judge scores and short justifications
- provenance copied from the original query result
- summary statistics such as exact-match rate and average judge scores

What just happened?

We created a reusable measurement loop that stays aligned with the real retrieval system.

Common beginner mistakes:

- evaluating only final answers and throwing away provenance
- letting one failed case crash the whole run instead of recording the error
- using only exact match and concluding the system is worse than it really is
- using only an LLM judge and forgetting that judge prompts can drift too

## 6. Make it runnable from the command line

This project also includes a real evaluation script.

```bash
python3 scripts/run_eval.py \
  --fixture evals/sample_eval_fixture.json \
  --output evals/report.json
```

The fixture contains documents plus cases. The script:

1. builds the normal Graph RAG service from `.env`
2. ingests the fixture documents
3. runs each evaluation case through the existing query path
4. writes a JSON report to disk

What just happened?

We turned evaluation into a repeatable artifact instead of a manual notebook exercise.

Design note:

The script still uses the real extractor, query answer model, and judge model, so it needs valid `.env` settings and an `OPENAI_API_KEY` for the real path. For an offline walkthrough, use the mocked demo in `examples/phase7_eval_demo.py`.

## 7. Why JSON output matters

JSON reports are not just convenient. They make it easy to:

- diff runs over time
- archive historical evals
- graph metrics later in a notebook or dashboard
- separate the measuring step from the interpretation step

If the script only printed text to the terminal, you would lose most of that leverage.

## Alternatives you should know about

Alternative 1: embedding similarity instead of exact match

This is useful for semantic comparison, but it can hide precise factual differences.

Alternative 2: human-only review

This is higher quality, but slower and harder to repeat frequently.

Alternative 3: task-specific structured scoring

For narrow domains, you can grade exact fields instead of full answers. That is often stronger than generic judging when your outputs have a clear schema.

## Experiment prompt

Run the same eval fixture twice.

- first with your normal local retrieval settings
- then with `top_k=1` for the local cases

Compare the exact-match rate and average faithfulness score. Did the system become more precise, more brittle, or both?

## Checkpoint question

If exact match goes down but judge-based relevance stays high, what does that suggest about the system's answers?

## What questions do you have before you start running this on your own corpus?
