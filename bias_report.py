"""Bias and calibration report on a code-preference dataset, written as Markdown.

Accuracy says how often an annotator is wrong, not how. This report shows the
direction of the errors, every number with its n:
- label bias: how often each label is picked, against how often the gold has it
- calibration: confidence against correctness
- confusion: which gold label gets answered as which

Usage: python bias_report.py [dataset], code-preference-v3 by default.
"""

import sys
from pathlib import Path

import pandas as pd

from responses import PREFERENCES, load_responses

DATASET = sys.argv[1] if len(sys.argv) > 1 else "code-preference-v3"
OUT = Path(__file__).parent / "out" / DATASET / "quality_report.md"
SMALL_N = 5


def md_table(headers, rows):
    """A Markdown table from a header list and a list of row lists."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def n_note(n):
    return f"{n} (small n)" if n < SMALL_N else str(n)


df, record_ids = load_responses(DATASET)
if df.empty:
    raise SystemExit(f"{DATASET}: no submitted answers on its {len(record_ids)} records, nothing to report")
annotators = sorted(df["annotator"].unique())
unanswered = len(set(record_ids) - set(df["record_id"]))
report = [
    f"# Annotation quality report: {DATASET}",
    "",
    f"{len(df)} answers on {len(record_ids)} records ({unanswered} unanswered), small n below {SMALL_N}.",
]

# Label bias: share picked minus share in the gold, over the records the annotator answered
bias_rows = []
for annotator in annotators:
    answers = df[df["annotator"] == annotator]
    picked = answers["preference"].value_counts(normalize=True).reindex(PREFERENCES, fill_value=0)
    expected = answers["key"].value_counts(normalize=True).reindex(PREFERENCES, fill_value=0)
    diff = (picked - expected) * 100
    bias_rows.append([annotator, n_note(len(answers)), *(f"{value:+.0f}" for value in diff)])
report += [
    "",
    "## Label bias",
    "",
    "Points picked minus points in the gold. `A` well above `B` with a balanced gold is position bias; "
    "`both_bad` below zero means real defects get waved through.",
    "",
    md_table(["annotator", "n", *PREFERENCES], bias_rows),
]

# Calibration: confidence against correctness
calibration_rows = []
for annotator in [*annotators, "all"]:
    answers = df if annotator == "all" else df[df["annotator"] == annotator]
    sure = answers["confidence"] == "sure"
    deferred = answers["confidence"].isin(["unsure", "cannot_judge"])
    right = answers["correct"]
    calibration_rows.append(
        [
            annotator,
            n_note(len(answers)),
            int((sure & right).sum()),
            int((sure & ~right).sum()),
            int((deferred & right).sum()),
            int((deferred & ~right).sum()),
            int(answers["confidence"].isna().sum()),
        ]
    )
report += [
    "",
    "## Calibration",
    "",
    "`sure wrong` reaches the training data unreviewed: the costly cell. "
    "`deferred right` costs reviewer time for nothing. "
    "`not asked` counts answers from a round without the confidence question.",
    "",
    md_table(
        ["annotator", "n", "sure right", "sure wrong", "deferred right", "deferred wrong", "not asked"],
        calibration_rows,
    ),
]

# Confusion: rows are the gold label, columns what annotators answered
confusion = pd.crosstab(df["key"], df["preference"]).reindex(index=PREFERENCES, columns=PREFERENCES, fill_value=0)
confusion_rows = [[f"gold {label}", n_note(int(confusion.loc[label].sum())), *confusion.loc[label]] for label in PREFERENCES]
report += [
    "",
    "## Confusion",
    "",
    "Off-diagonal cells are the systematic mistakes. A whole row off the diagonal points at a guideline gap.",
    "",
    md_table(["gold", "n", *(f"answered {label}" for label in PREFERENCES)], confusion_rows),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(report) + "\n")
print("\n".join(report))
print(f"\nwritten to {OUT}")
