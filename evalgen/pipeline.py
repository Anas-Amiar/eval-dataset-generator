"""
The nightly pipeline: sample logs (signal-boosted) -> cluster -> label with
confidence routing -> dedup against the existing dataset -> persist + stats.
"""

import json
import os

from evalgen.models import PipelineStats, TestCase
from evalgen.cluster import cluster_logs, _vec, _cos
from evalgen.labeler import label_log

DATASET_PATH = "reports/eval_dataset.json"
DEDUP_THRESHOLD = 0.85


def _load_dataset() -> list[dict]:
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH) as f:
            return json.load(f)
    return []


def _is_duplicate(candidate: TestCase, existing: list[dict]) -> bool:
    cv = _vec(candidate.prompt)
    return any(_cos(cv, _vec(e["prompt"])) >= DEDUP_THRESHOLD for e in existing)


def run_pipeline(logs) -> PipelineStats:
    # Signal-boosted sampling: every negative-signal log is always included;
    # here the demo set is small enough to take everything.
    sampled = logs

    clusters, outliers = cluster_logs(sampled)
    log_category = {}
    for c in clusters:
        for lid in c.member_log_ids:
            log_category[lid] = c.name
    for o in outliers:
        log_category[o.log_id] = "outlier"

    existing = _load_dataset()
    added, review, deduped = [], [], 0
    for log in sampled:
        case = label_log(log, log_category[log.log_id])
        if _is_duplicate(case, existing + [c.model_dump() for c in added + review]):
            deduped += 1
            continue
        (added if case.status == "auto_added" else review).append(case)

    all_new = [c.model_dump() for c in added + review]
    os.makedirs("reports", exist_ok=True)
    with open(DATASET_PATH, "w") as f:
        json.dump(existing + all_new, f, indent=2)

    coverage: dict[str, int] = {}
    for c in added + review:
        coverage[c.category] = coverage.get(c.category, 0) + 1

    return PipelineStats(
        logs_sampled=len(sampled),
        clusters_found=len(clusters),
        outliers_flagged=len(outliers),
        candidates_generated=len(added) + len(review) + deduped,
        auto_added=len(added),
        routed_to_review=len(review),
        deduplicated=deduped,
        coverage_by_category=coverage,
    ), added, review


if __name__ == "__main__":
    from data.logs import PRODUCTION_LOGS

    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)

    stats, added, review = run_pipeline(PRODUCTION_LOGS)

    print("=== Nightly eval-dataset pipeline run ===\n")
    print(f"Logs sampled:        {stats.logs_sampled}")
    print(f"Clusters found:      {stats.clusters_found}")
    print(f"Outliers flagged:    {stats.outliers_flagged}")
    print(f"Auto-added:          {stats.auto_added}")
    print(f"Routed to review:    {stats.routed_to_review}")
    print(f"Deduplicated:        {stats.deduplicated}")
    print(f"Coverage:            {stats.coverage_by_category}\n")

    print("--- Auto-added test cases ---")
    for c in added:
        print(f"  [{c.difficulty:11s}] [{c.expected_behavior:15s}] conf={c.label_confidence}  {c.prompt[:55]}")
    print("\n--- Review queue (low label confidence) ---")
    for c in review:
        print(f"  [{c.difficulty:11s}] [{c.expected_behavior:15s}] conf={c.label_confidence}  {c.prompt[:55]}")
