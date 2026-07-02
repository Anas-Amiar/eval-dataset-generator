"""
Interaction clustering: greedy centroid clustering over bag-of-words vectors
(production swap: real embeddings + HDBSCAN — the pipeline shape is identical).
Logs that don't fit any cluster are OUTLIERS — and outliers are the most
valuable eval candidates: novel requests, gibberish, adversarial probes.
"""

import math
import re

from evalgen.models import LogEntry, Cluster

CLUSTER_THRESHOLD = 0.18

STOP = {"the", "my", "i", "how", "do", "can", "is", "a", "an", "to", "for",
        "of", "and", "on", "in", "me", "it", "this", "that", "why", "was",
        "when", "where", "your", "you", "get", "all"}


def _vec(text: str) -> dict[str, float]:
    tokens = [t for t in re.findall(r"[a-z]+", text.lower())
              if t not in STOP and len(t) > 2]
    v: dict[str, float] = {}
    for t in tokens:
        v[t] = v.get(t, 0) + 1
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / n for t, x in v.items()}


def _cos(a: dict, b: dict) -> float:
    return sum(a.get(t, 0.0) * x for t, x in b.items())


def cluster_logs(logs: list[LogEntry]) -> tuple[list[Cluster], list[LogEntry]]:
    clusters: list[dict] = []   # {"centroid": vec, "members": [log]}
    outliers: list[LogEntry] = []

    for log in logs:
        v = _vec(log.prompt)
        best_sim, best = 0.0, None
        for c in clusters:
            sim = _cos(v, c["centroid"])
            if sim > best_sim:
                best_sim, best = sim, c
        if best is not None and best_sim >= CLUSTER_THRESHOLD:
            best["members"].append(log)
            # update centroid (mean of member vectors, renormalized-ish)
            for t, x in v.items():
                best["centroid"][t] = best["centroid"].get(t, 0.0) + x / len(best["members"])
        else:
            clusters.append({"centroid": dict(v), "members": [log]})

    named: list[Cluster] = []
    real_clusters = [c for c in clusters if len(c["members"]) >= 2]
    for i, c in enumerate(real_clusters):
        # name from the most frequent content word among members
        counts: dict[str, int] = {}
        for m in c["members"]:
            for t in _vec(m.prompt):
                counts[t] = counts.get(t, 0) + 1
        name = max(counts, key=counts.get)
        named.append(Cluster(cluster_id=i, name=name,
                             member_log_ids=[m.log_id for m in c["members"]]))
    for c in clusters:
        if len(c["members"]) < 2:
            outliers.extend(c["members"])
    return named, outliers


if __name__ == "__main__":
    from data.logs import PRODUCTION_LOGS

    clusters, outliers = cluster_logs(PRODUCTION_LOGS)
    for c in clusters:
        print(f"cluster '{c.name}': {c.member_log_ids}")
    print(f"outliers: {[o.log_id for o in outliers]}")
