"""The SWE-bench Verified rubric (OpenAI, 2024): is the issue well specified, do the tests accept every valid fix.

Three annotators per sample. A sample is dropped when any annotator rates either scale 2 or more,
or reports another major issue: the most severe label wins.
"""

import great_expectations as gx
import pandas as pd

from argilla_lab.core.rubric import Question, Rubric

SCALE = (0, 1, 2, 3)
FAST_MINUTES = 5


def prepare(answers):
    answers = answers.copy()
    claimed = pd.to_datetime(answers["claimed_at"], utc=True, format="mixed")
    submitted = pd.to_datetime(answers["submitted_at"], utc=True, format="mixed")
    answers["minutes"] = (submitted - claimed).dt.total_seconds() / 60
    return answers


RUBRIC = Rubric(
    name="swe-verified",
    questions=(
        Question("underspecified", SCALE, ordinal=True, drop_at=2, notes="underspecified_notes"),
        Question("false_negative", SCALE, ordinal=True, drop_at=2, notes="false_negative_notes"),
        Question("other_major_issues", (0, 1), ordinal=True, drop_at=1),
    ),
    aggregation="max",
    min_votes=3,
    min_notes=100,
    confidence="annotator_confidence",
    confidence_labels=(1, 2, 3, 4, 5),
    fields=("claimed_at", "submitted_at"),
    prepare=prepare,
    answer_checks=(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="minutes", min_value=0, description="Submitted after it was claimed"
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="minutes",
            min_value=FAST_MINUTES,
            description=f"Not submitted within {FAST_MINUTES} minutes of claiming (issue, patch and tests to read)",
        ),
    ),
    min_profile=10,
)
