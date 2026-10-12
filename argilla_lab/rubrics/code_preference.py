"""The code-preference rubric: pick the better of two answers, tag the issues, say how sure you are.

The guideline numbers refer to the guidelines in annotate/create_dataset.py.
"""

import great_expectations as gx

from argilla_lab.core.rubric import Question, Rubric

PREFERENCES = ("A", "B", "tie", "both_bad")
SUBSTANTIVE = {"bug", "security", "ignores_instructions", "performance"}


def issues_kind(issues):
    """What the issue tags claim: empty, none, style_only, or substantive."""
    tags = set(issues) if isinstance(issues, list) else set()
    if not tags:
        return "empty"
    if tags & SUBSTANTIVE:
        return "substantive"
    return "none" if tags == {"none"} else "style_only"


def prepare(answers):
    answers = answers.copy()
    answers["issues_kind"] = answers["issues"].map(issues_kind)
    return answers


# Guideline 2: tie means style only. A, B or both_bad needs a substantive issue.
# An empty tag list is left to the guideline 4 check.
CONSISTENT = [(p, "empty") for p in PREFERENCES] + [
    ("A", "substantive"),
    ("B", "substantive"),
    ("both_bad", "substantive"),
    ("tie", "style_only"),
    ("tie", "none"),
]

RUBRIC = Rubric(
    name="code-preference",
    questions=(Question("preference", PREFERENCES, notes="rationale", gold=True),),
    aggregation="majority",
    min_votes=2,
    min_notes=20,
    confidence="confidence",
    confidence_labels=("sure", "unsure", "cannot_judge"),
    sure=("sure",),
    fields=("issues",),
    prepare=prepare,
    answer_checks=(
        gx.expectations.ExpectColumnValuesToNotBeInSet(
            column="issues_kind", value_set=["empty"], description="Guideline 4: at least one issue tag, none counts"
        ),
        gx.expectations.ExpectColumnPairValuesToBeInSet(
            column_A="preference",
            column_B="issues_kind",
            value_pairs_set=CONSISTENT,
            description="Guideline 2: the preference matches the issue tags",
        ),
    ),
)
