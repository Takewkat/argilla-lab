"""Quality report on an Argilla dataset, as Markdown: decisions, agreement, annotators, and with a gold key
label bias, calibration and confusion. Sections follow the dataset's rubric (argilla_lab/rubrics/).

Usage: uv run lab report [dataset], code-preference-v3 by default.
"""

import sys

from argilla_lab.core import checks, report
from argilla_lab.core.paths import OUT_DIR
from argilla_lab.core.responses import load_responses
from argilla_lab.rubrics import for_dataset

DATASET = sys.argv[1] if len(sys.argv) > 1 else "code-preference-v3"
OUT = OUT_DIR / DATASET / "quality_report.md"

rubric = for_dataset(DATASET)
answers, record_ids = load_responses(DATASET, rubric)
if answers.empty:
    raise SystemExit(f"{DATASET}: no submitted answers on its {len(record_ids)} records, nothing to report")
evaluation = checks.evaluate(rubric, answers, record_ids)
print("\n".join(report.write(rubric, evaluation, f"Annotation quality report: {DATASET}", OUT)))
print(f"\nwritten to {OUT}")
