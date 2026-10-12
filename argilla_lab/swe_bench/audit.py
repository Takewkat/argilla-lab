"""Quality audit of the SWE-bench Verified human annotations, with the lab's own checks and report.

OpenAI published the labels, not the checks on the labellers. This runs the three layers of
core/checks.py under the swe-verified rubric (argilla_lab/rubrics/swe_verified.py), first
reproduces the published keep/drop decision, then writes the report. An audit, not a gate:
it always exits 0.

Usage: uv run lab swe-audit
"""

from argilla_lab.core import checks, report
from argilla_lab.core.paths import OUT_DIR
from argilla_lab.rubrics.swe_verified import RUBRIC
from argilla_lab.swe_bench.annotations import answers, load

OUT = OUT_DIR / "swe-bench-verified" / "quality_report.md"

per_annotator, ensembled = load()
frame, record_ids = answers(per_annotator)
evaluation = checks.evaluate(RUBRIC, frame, record_ids)
print(f"SWE-bench Verified annotations: {len(frame)} answers, {len(evaluation.annotators)} annotators, "
      f"{len(record_ids)} samples")
print(f"counted votes: {checks.counting_rule(RUBRIC)}")
failed = checks.validate(RUBRIC, evaluation, show=5)

# Our decision against the published filter_out, sample by sample
published = ensembled.set_index("instance_id")["filter_out"].astype(str).eq("True")
ours = evaluation.records.set_index("record_id")["dropped"].reindex(published.index)
reproduced = f"Reproduces the published keep/drop decision on {int((ours == published).sum())} of {len(published)} samples."

lines = report.write(
    RUBRIC, evaluation, "SWE-bench Verified annotations: quality audit", OUT, failed=failed, notes=[reproduced]
)
print("\n" + "\n".join(lines[:4]))
print(f"\nwritten to {OUT}")
