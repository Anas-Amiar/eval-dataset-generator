from pydantic import BaseModel
from typing import Literal, Optional


class LogEntry(BaseModel):
    log_id: str
    feature: str                     # which app feature made this LLM call
    prompt: str
    response: str
    latency_ms: float
    user_feedback: Optional[Literal["thumbs_up", "thumbs_down"]] = None
    was_retried: bool = False        # user immediately rephrased and asked again


class Cluster(BaseModel):
    cluster_id: int
    name: str                        # named from its representative example
    member_log_ids: list[str]


class TestCase(BaseModel):
    case_id: str
    source_log_id: str
    prompt: str
    category: str                    # cluster name or "outlier"
    difficulty: Literal["simple", "moderate", "hard", "adversarial"]
    expected_behavior: Literal["should_answer", "should_refuse", "should_clarify"]
    quality_score: float             # judged quality of the production response (1-5)
    must_contain: list[str]
    must_not_contain: list[str]
    label_confidence: float          # agreement across labeling runs
    status: Literal["auto_added", "needs_review"]


class PipelineStats(BaseModel):
    logs_sampled: int
    clusters_found: int
    outliers_flagged: int
    candidates_generated: int
    auto_added: int
    routed_to_review: int
    deduplicated: int
    coverage_by_category: dict[str, int]
