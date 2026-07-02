# Automated Eval Dataset Generator from Production Logs

A pipeline that mines production LLM logs and automatically converts them into labeled
eval test cases — clustering traffic into natural categories, flagging outliers (the most
valuable candidates), auto-labeling with confidence-based routing, and deduplicating
against the existing dataset. The eval suite grows nightly while you sleep.

Demo run over 15 production logs from a support assistant:

```
Clusters found:    4  (billing/charges, plans, app crashes, account/email)
Outliers flagged:  6  — including the three that matter most:
                      prompt injection  -> adversarial / should_refuse
                      gibberish input   -> hard / should_clarify
                      off-topic probe   -> should_refuse
Auto-added:        11 test cases (label confidence >= 0.8)
Routed to review:   4 (labeling runs disagreed — a human decides)
```

## Why this exists

The hardest part of AI evaluation isn't the eval harness — it's the dataset. Hand-curated
golden sets go stale within weeks, and they never contain the weird things real users
actually type. Production traffic is an endless supply of exactly the right test cases:
the questions users really ask, the failures that really happened (thumbs-down + retry
signals), and the adversarial probes someone really attempted. This pipeline turns that
supply into labeled eval data automatically.

## How it works

```
evalgen/
  models.py    Typed shapes: LogEntry, Cluster, TestCase, PipelineStats
  cluster.py   Greedy centroid clustering over BOW vectors (production swap:
               embeddings + HDBSCAN). Logs that fit no cluster are OUTLIERS —
               novel requests, gibberish, injection attempts
  labeler.py   Auto-labeling: quality judged from feedback signals (thumbs-down +
               retry = failure case), difficulty estimation, expected-behavior
               classification (answer/refuse/clarify), assertion generation —
               and CONFIDENCE ROUTING: 3 noisy labeling runs; high agreement ->
               auto-add, low agreement -> human review queue
  pipeline.py  The nightly job: sample -> cluster -> label -> dedup (cosine >=
               0.85 against existing cases) -> persist + coverage stats
data/logs.py   15 synthetic production logs with the signals that matter:
               thumbs-up/down, retries, an injection attempt, gibberish, off-topic
reports/       The growing eval dataset (gitignored)
```

### What each log becomes

A `TestCase` with: category (from its cluster), difficulty (simple/moderate/hard/
adversarial), expected behavior (should_answer / should_refuse / should_clarify),
a judged quality score for the production response (low scores become negative test
cases — "verify future models fix this"), must-contain / must-not-contain assertions,
and a label confidence that decides auto-add vs. review.

## Setup

```bash
git clone https://github.com/Anas-Amiar/Project-14-eval-dataset-generator.git
cd "Project 14 - eval-dataset-generator"
pip install -r requirements.txt

python3 -m evalgen.cluster    # see traffic clustered + outliers surfaced
python3 -m evalgen.pipeline   # the full nightly run with stats
```

## Architecture decisions

**Why are outliers the most valuable candidates?**
Clustered traffic is what you already know your system does. Outliers are what you
didn't anticipate: the injection attempt, the gibberish, the off-topic probe, the novel
question. Those are exactly the cases a hand-curated golden set never contains — and
exactly what the next model version needs to be tested against.

**Why 3 labeling runs instead of 1?**
LLM judges are non-deterministic. Running the labeler multiple times and measuring
agreement converts that non-determinism into a confidence signal: unanimous runs
auto-add, disagreeing runs go to a human. The dataset grows fast without silently
accumulating garbage labels — the same confidence-routing pattern as my OCR review queue.

**Why do thumbs-down + retry signals set quality low instead of excluding the log?**
A production failure is a *perfect* test case — as a negative example. "The model used to
answer this badly" becomes "verify the next model answers it well." Excluding failures
would throw away the most informative data in the logs.

**Why dedup against the existing dataset?**
Users ask the same question a thousand ways; without dedup the dataset becomes 90%
"how do I get a refund" and coverage metrics lie. Near-duplicate candidates (cosine ≥
0.85) are skipped, and the coverage-by-category stats show where the dataset is thin.

## What's deliberately out of scope for v1

- Real embeddings + HDBSCAN (the greedy clusterer isolates the same interface)
- Real LLM-as-judge labeling (the noisy-runs + agreement mechanism is in place)
- Golden answer / rubric generation per test case
- The eval runner + regression tracker (see Project 1, which is exactly that)
- PII redaction rules on ingestion
- The curation dashboard for the review queue
