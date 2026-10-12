"""Quality audit of the SWE-bench Verified human annotations, with the lab's three-layer checks.

OpenAI published the labels, not the checks on the labellers. This runs them:
- answers: each annotation against the published rubric (labels in range, notes of 100+ chars)
- annotators: severity against the peers who rated the same samples
- samples: three votes each, and whether the keep/drop decision depends on the aggregation rule

Then agreement on the 0-3 scales, nominal against interval Krippendorff's alpha, and a
Markdown report in out/swe-audit/report.md. An audit, not a gate: it always exits 0.
"""

import great_expectations as gx
import pandas as pd

from argilla_lab.core.agreement import interval, krippendorff_alpha, nominal
from argilla_lab.core.gx_runner import validate
from argilla_lab.core.paths import OUT_DIR
from argilla_lab.swe_bench.annotations import load

OUT = OUT_DIR / "swe-audit" / "report.md"
SCALED = ["underspecified", "false_negative"]
MIN_NOTES = 100
MIN_PROFILE = 10
MAX_PEER_BIAS = 0.75
FAST_MINUTES = 5

per, ensembled = load()
per["minutes"] = (
    pd.to_datetime(per["submitted_at"], utc=True, format="mixed")
    - pd.to_datetime(per["claimed_at"], utc=True, format="mixed")
).dt.total_seconds() / 60

# The published rule: an annotator flags a sample when a scaled label is 2+ or a major issue is reported
per["flags"] = (per["underspecified"] >= 2) | (per["false_negative"] >= 2) | (per["other_major_issues"] >= 1)
for column in SCALED:
    group = per.groupby("instance_id")[column]
    peers_mean = (group.transform("sum") - per[column]) / (group.transform("count") - 1)
    per[f"{column}_vs_peers"] = per[column] - peers_mean

samples = per.groupby("instance_id").agg(
    flags=("flags", "sum"),
    underspecified_votes=("underspecified", "count"),
    false_negative_votes=("false_negative", "count"),
    underspecified_median=("underspecified", "median"),
    false_negative_median=("false_negative", "median"),
    other_median=("other_major_issues", "median"),
)
samples["drop_max"] = samples["flags"] >= 1
samples["drop_median"] = (
    (samples["underspecified_median"] >= 2) | (samples["false_negative_median"] >= 2) | (samples["other_median"] >= 1)
)
samples["same_decision"] = samples["drop_max"] == samples["drop_median"]
samples = samples.reset_index()

# Who alone turned a sample into a drop: the only flag among the three
sole = per[per["instance_id"].isin(samples.loc[samples["flags"] == 1, "instance_id"]) & per["flags"]]
annotators = per.groupby("user_id").agg(
    answers=("instance_id", "size"),
    underspecified_vs_peers=("underspecified_vs_peers", "mean"),
    false_negative_vs_peers=("false_negative_vs_peers", "mean"),
    confidence=("annotator_confidence", "mean"),
    median_minutes=("minutes", "median"),
)
annotators["sole_drops"] = sole.groupby("user_id").size().reindex(annotators.index, fill_value=0)
annotators = annotators.reset_index()

answers_suite = gx.ExpectationSuite(
    name="swe_answers",
    expectations=[
        gx.expectations.ExpectCompoundColumnsToBeUnique(
            column_list=["instance_id", "user_id"], description="One annotation per annotator and sample"
        ),
        *(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column, description=f"{column} is answered")
            for column in [*SCALED, "other_major_issues"]
        ),
        *(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=column, value_set=[0, 1, 2, 3], description=f"{column} on the 0-3 scale"
            )
            for column in SCALED
        ),
        *(
            gx.expectations.ExpectColumnValueLengthsToBeBetween(
                column=f"{column}_notes",
                min_value=MIN_NOTES,
                description=f"Rubric: {column} explained in {MIN_NOTES}+ chars (null notes count as missing below)",
            )
            for column in SCALED
        ),
        *(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=f"{column}_notes", description=f"Rubric: {column} explanation present"
            )
            for column in SCALED
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="annotator_confidence", min_value=1, max_value=5, description="Confidence on the 1-5 scale"
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="minutes", min_value=0, description="Submitted after it was claimed"
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="minutes",
            min_value=FAST_MINUTES,
            description=f"Not submitted within {FAST_MINUTES} minutes of claiming (issue, patch and tests to read)",
        ),
    ],
)

annotators_suite = gx.ExpectationSuite(
    name="swe_annotators",
    expectations=[
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="answers", min_value=MIN_PROFILE, description=f"{MIN_PROFILE}+ annotations, enough to profile"
        ),
        *(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=f"{column}_vs_peers",
                min_value=-MAX_PEER_BIAS,
                max_value=MAX_PEER_BIAS,
                description=f"{column}: within {MAX_PEER_BIAS} points of peers on the same samples",
            )
            for column in SCALED
        ),
    ],
)

samples_suite = gx.ExpectationSuite(
    name="swe_samples",
    expectations=[
        *(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=f"{column}_votes", min_value=3, description=f"{column}: three votes, not two"
            )
            for column in SCALED
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="same_decision",
            value_set=[True],
            description="Keep/drop does not depend on the rule (max severity vs median)",
        ),
    ],
)


print(f"SWE-bench Verified annotations: {len(per)} answers, {len(annotators)} annotators, {len(samples)} samples")
failed = [
    *validate(answers_suite, per, ["instance_id", "user_id"], show=5),
    *validate(annotators_suite, annotators, ["user_id"], show=10),
    *validate(samples_suite, samples, ["instance_id"], show=5),
]

alphas = {
    column: {
        name: krippendorff_alpha([g.to_numpy(float) for _, g in per.groupby("instance_id")[column]], metric)
        for name, metric in (("nominal", nominal), ("interval", interval))
    }
    for column in SCALED
}
flipped = samples[~samples["same_decision"]]
top_sole = annotators.sort_values("sole_drops", ascending=False).head(5)
short_notes = {c: int((per[f"{c}_notes"].fillna("").str.len() < MIN_NOTES).sum()) for c in SCALED}

report = [
    "# SWE-bench Verified annotations: quality audit",
    "",
    f"{len(per)} annotations of {len(samples)} samples by {len(annotators)} annotators.",
    f"Dropped by the published rule (max severity): {samples['drop_max'].sum()} ({samples['drop_max'].mean():.1%}).",
    f"Dropped by the median of the three: {samples['drop_median'].sum()} ({samples['drop_median'].mean():.1%}).",
    f"Decision depends on the rule: {len(flipped)} samples, {int((flipped['flags'] == 1).sum())} dropped on a single "
    f"annotator's flag, {int((flipped['flags'] > 1).sum())} on flags for different criteria.",
    "",
    "## Agreement, Krippendorff's alpha",
    "",
    "| label | nominal | interval |",
    "|---|---|---|",
    *(f"| {c} | {a['nominal']:.2f} | {a['interval']:.2f} |" for c, a in alphas.items()),
    "",
    "Nominal treats 1 vs 2 like 0 vs 3; interval weighs the distance on the scale.",
    "",
    "## Rubric compliance",
    "",
    *(f"- {c}: {n} explanations shorter than {MIN_NOTES} chars or missing" for c, n in short_notes.items()),
    f"- {int((per['minutes'] < FAST_MINUTES).sum())} annotations submitted within {FAST_MINUTES} minutes of claiming",
    "",
    "## Annotators who most often dropped a sample alone",
    "",
    "| annotator | annotations | sole drops | underspecified vs peers | false_negative vs peers |",
    "|---|---|---|---|---|",
    *(
        f"| {r.user_id} | {r.answers} | {r.sole_drops} | {r.underspecified_vs_peers:+.2f} | {r.false_negative_vs_peers:+.2f} |"
        for r in top_sole.itertuples()
    ),
    "",
    f"Checks failed: {len(failed)}",
    *(f"- {name}" for name in failed),
]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(report) + "\n")
print("\n" + "\n".join(report[:7]))
print(f"\nwritten to {OUT}")
