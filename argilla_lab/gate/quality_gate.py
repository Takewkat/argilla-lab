"""Quality gate on an Argilla dataset, to run before lab export.

Runs the three layers of core/checks.py with the dataset's rubric (argilla_lab/rubrics/):
- answers: each answer against the rubric
- annotators: can this annotator be trusted
- records: is this record ready, counting only the votes the rubric counts

Exits 1 when any check fails, so nothing downstream runs on data that failed the gate.
Moves the round to gated or gate_failed in rounds.json; a gated round keeps a fingerprint
of the answers it checked, so the export can refuse answers changed since.

Usage: uv run lab gate [dataset], code-preference-v3 by default.
"""

import sys

from argilla_lab.core import checks, rounds
from argilla_lab.core.responses import fingerprint, load_responses
from argilla_lab.rubrics import for_dataset

DATASET = sys.argv[1] if len(sys.argv) > 1 else "code-preference-v3"
GATEABLE = {"annotating", "gate_failed", "gated"}

status = rounds.get(DATASET)["status"]
if status == "canceled":
    print(f"{DATASET} is canceled, nothing to gate")
    raise SystemExit(0)
if status not in GATEABLE:
    hint = f": run uv run lab rounds start {DATASET}" if status is None else ""
    raise SystemExit(f"{DATASET} is {status or 'unregistered'}, the gate runs on {', '.join(sorted(GATEABLE))}{hint}")

rubric = for_dataset(DATASET)
answers, record_ids = load_responses(DATASET, rubric)
evaluation = checks.evaluate(rubric, answers, record_ids)
print(f"{DATASET} ({rubric.name} rubric): {len(answers)} answers, {len(evaluation.annotators)} annotators, "
      f"{len(record_ids)} records")
print(f"counted votes: {checks.counting_rule(rubric)}")
failed = checks.validate(rubric, evaluation)
print(f"\ntrusted annotators: {', '.join(evaluation.trusted) or 'none'}\n")
if failed:
    rounds.move(DATASET, "gate_failed", f"{len(failed)} checks failed: " + "; ".join(failed))
    raise SystemExit(f"gate FAILED on {DATASET}: do not export")
rounds.move(DATASET, "gated", "all checks passed", fingerprint=fingerprint(answers, rubric))
