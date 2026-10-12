"""Quality gate on a code-preference dataset, to run before lab export.

Validates the submitted answers with Great Expectations on three layers:
- responses: one row per annotator and record, one expectation per annotation guideline
- annotators: can this annotator be trusted, measured on the gold answers
- records: is this record ready for training, counting trusted votes only

Exits 1 when any check fails, so nothing downstream runs on data that failed the gate.
Moves the round to gated or gate_failed in rounds.json; a gated round keeps a fingerprint
of the answers it checked, so the export can refuse answers changed since.
"""

import sys

import great_expectations as gx
import pandas as pd

from argilla_lab.core import rounds
from argilla_lab.core.gx_runner import validate
from argilla_lab.core.responses import PREFERENCES, fingerprint, load_responses

DATASET = sys.argv[1] if len(sys.argv) > 1 else "code-preference-v3"
CONFIDENCES = ["sure", "unsure", "cannot_judge"]
PLACEHOLDERS = [r"^\s*$", r"(?i)^\s*(test|n/?a|none|todo|asdf|-+|\.+)\s*$"]
MIN_RATIONALE = 20
MIN_ANNOTATORS = 2
MIN_SURE = 3
MIN_GOLD_ACCURACY = 0.6
MAX_DEFERRED = 0.5
MIN_TRUSTED_VOTES = 2

GATEABLE = {"annotating", "gate_failed", "gated"}

status = rounds.get(DATASET)["status"]
if status == "canceled":
    print(f"{DATASET} is canceled, nothing to gate")
    raise SystemExit(0)
if status not in GATEABLE:
    hint = f": run uv run lab rounds start {DATASET}" if status is None else ""
    raise SystemExit(f"{DATASET} is {status or 'unregistered'}, the gate runs on {', '.join(sorted(GATEABLE))}{hint}")

df, record_ids = load_responses(DATASET)

# Annotator layer: accuracy is measured on sure answers only, an unsure answer is not an error
sure = df[df["confidence"] == "sure"]
annotators = (
    df.groupby("annotator")
    .agg(answers=("record_id", "size"))
    .join(sure.groupby("annotator").agg(sure=("record_id", "size"), gold_accuracy=("correct", "mean")))
    .fillna({"sure": 0})
    .reset_index()
)
annotators["deferred_rate"] = 1 - annotators["sure"] / annotators["answers"]
trusted = annotators.loc[
    (annotators["sure"] >= MIN_SURE) & (annotators["gold_accuracy"] >= MIN_GOLD_ACCURACY), "annotator"
]

# Record layer: only sure votes from trusted annotators count toward a training label
votes = sure[sure["annotator"].isin(trusted)].groupby("record_id")["preference"]
records = (
    pd.DataFrame({"record_id": record_ids})
    .merge(sure.groupby("record_id").size().rename("sure_votes"), on="record_id", how="left")
    .merge(
        votes.agg(trusted_votes="size", majority_share=lambda s: s.value_counts().iloc[0] / len(s)),
        on="record_id",
        how="left",
    )
    .fillna({"sure_votes": 0, "trusted_votes": 0})
)

# Guideline 2: tie means style only. A, B or both_bad needs a substantive issue.
# An empty tag list is left to the guideline 4 check.
CONSISTENT = [(p, "empty") for p in PREFERENCES] + [
    ("A", "substantive"),
    ("B", "substantive"),
    ("both_bad", "substantive"),
    ("tie", "style_only"),
    ("tie", "none"),
]

responses_suite = gx.ExpectationSuite(
    name="responses",
    expectations=[
        # Every other check holds on an empty table, so emptiness is a check of its own
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1, description="At least one submitted answer"),
        gx.expectations.ExpectCompoundColumnsToBeUnique(
            column_list=["record_id", "annotator"],
            description="One submission per annotator and record",
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="key",
            description="Every record has a gold answer in answer_key.json",
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="preference", value_set=PREFERENCES, description="Guideline 1: a valid preference label"
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="confidence", value_set=CONFIDENCES, description="Guideline 6: a valid confidence label"
        ),
        # The in-set checks skip nulls: a round created before a question existed passes them empty
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="preference", description="Guideline 1: every answer has a preference"
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="confidence", description="Guideline 6: every answer has a confidence (asked since v2)"
        ),
        gx.expectations.ExpectColumnValuesToNotMatchRegexList(
            column="rationale", regex_list=PLACEHOLDERS, description="Guideline 5: no placeholder rationale"
        ),
        gx.expectations.ExpectColumnValueLengthsToBeBetween(
            column="rationale",
            min_value=MIN_RATIONALE,
            description=f"Guideline 5: a rationale long enough to name a line ({MIN_RATIONALE}+ chars)",
        ),
        gx.expectations.ExpectColumnValuesToNotBeInSet(
            column="issues_kind", value_set=["empty"], description="Guideline 4: at least one issue tag, none counts"
        ),
        gx.expectations.ExpectColumnPairValuesToBeInSet(
            column_A="preference",
            column_B="issues_kind",
            value_pairs_set=CONSISTENT,
            description="Guideline 2: the preference matches the issue tags",
        ),
    ],
)

annotators_suite = gx.ExpectationSuite(
    name="annotators",
    expectations=[
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=MIN_ANNOTATORS, description=f"{MIN_ANNOTATORS}+ annotators, agreement needs a second opinion"
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="sure",
            min_value=MIN_SURE,
            description=f"Enough sure answers to measure accuracy ({MIN_SURE}+)",
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="gold_accuracy",
            min_value=MIN_GOLD_ACCURACY,
            description=f"Sure answers match the gold key ({MIN_GOLD_ACCURACY:.0%}+, chance is 25%)",
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="deferred_rate",
            max_value=MAX_DEFERRED,
            description=f"At most {MAX_DEFERRED:.0%} of answers deferred to the reviewer",
        ),
    ],
)

records_suite = gx.ExpectationSuite(
    name="records",
    expectations=[
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1, description="The dataset has records"),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="trusted_votes",
            min_value=MIN_TRUSTED_VOTES,
            description=f"{MIN_TRUSTED_VOTES}+ sure votes from trusted annotators",
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="majority_share",
            min_value=0.5,
            strict_min=True,
            description="A strict majority among the trusted votes",
        ),
    ],
)

print(f"{DATASET}: {len(df)} answers, {len(annotators)} annotators, {len(records)} records")
failed = [
    *validate(responses_suite, df, ["record_id", "annotator"]),
    *validate(annotators_suite, annotators, ["annotator"]),
    *validate(records_suite, records, ["record_id"]),
]
print(f"\ntrusted annotators: {', '.join(trusted) or 'none'}\n")
if failed:
    rounds.move(DATASET, "gate_failed", f"{len(failed)} checks failed: " + "; ".join(failed))
    raise SystemExit(f"gate FAILED on {DATASET}: do not export")
rounds.move(DATASET, "gated", "all checks passed", fingerprint=fingerprint(df))
